import json
import torch
import pickle
import numpy as np
import random
import pandas as pd
import time
from tqdm import tqdm
#from modules import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error
from torch.utils.data import random_split,Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.masked import masked_tensor # NOTE - not using masked tensors anymore. they don't really work
from torch.utils.data import DataLoader


# Released CONDOR/AQA label definitions recovered from the paper and released data.
CONDOR_POWER_COST_COEFFICIENT = 3e-4
CONDOR_QOS_BETA = 0.8
CONDOR_QOS_RHO = 60.0
CONDOR_QOS_THRESHOLD = 0.1


class DCDataset(Dataset):
    # CONDOR labels and model inputs, populated from the FlexDC results CSV.
    def __init__(self,
                 data_file_path = 'traditional_iso16_fullpilot_AQA_combined_grid_search_results.csv',
                 use_norm_pr = True, # same default as released CONDOR: use normalized P and R
                 use_norm_cost = True, # same released CONDOR output scaling
                 use_norm_wlmix = True, # normalize workload features by their FlexDC dataset averages
                 pad_wlmix = True
                 ):
        csv = pd.read_csv(data_file_path)
        self.len = csv.shape[0]

        ### Cost - our labels!
        # Keep the original CONDOR component meanings:
        #   cost_power = 0.0003 * (P - R)
        #   cost_error = mean absolute tracking error in kW
        #   cost_qos   = 0.8 * sum_j SoftPlus(60 * (P_delay_j - 0.1))
        p_watts = csv['P_actual_watts'].to_numpy(dtype=float)
        r_watts = csv['R_actual_watts'].to_numpy(dtype=float)
        c_power = CONDOR_POWER_COST_COEFFICIENT * (p_watts - r_watts)
        c_error = csv['Mtrack_Error_MeanAbs_Watts'].to_numpy(dtype=float) / 1000.0

        qos_probabilities = [np.asarray(json.loads(item), dtype=float) for item in csv['QoS_Delay_Probabilities']]
        c_qos = np.asarray([
            CONDOR_QOS_BETA * np.logaddexp(
                0,
                CONDOR_QOS_RHO * (probabilities - CONDOR_QOS_THRESHOLD),
            ).sum()
            for probabilities in qos_probabilities
        ])

        if use_norm_cost:
            server_counts = csv['server_count'].to_numpy(dtype=float)
            job_mix_length = csv['workload_mix_size'].to_numpy(dtype=float)
            # Preserve the released CONDOR implementation scaling exactly.
            c_power,c_error,c_qos = (c_power*120)/server_counts,(c_error*200)/server_counts,c_qos/job_mix_length

        costs = np.zeros((self.len,3))
        costs[:,0],costs[:,1],costs[:,2] = c_power,c_error,c_qos
        self.costs = torch.tensor(costs, dtype=torch.float)

        ### Our tabular server features
        if use_norm_pr:
            p,r = csv['Pbar_ratio'].to_numpy(dtype=float),csv['R_ratio'].to_numpy(dtype=float)
        else:
            # Same behavior as released CONDOR: total watts -> total kW.
            p,r = p_watts/1000.0,r_watts/1000.0
        client_count = csv['server_count'].to_numpy(dtype=float)
        util = csv['utilization'].to_numpy(dtype=float)
        wload_size = csv['workload_mix_size'].to_numpy(dtype=float)
        feats = np.zeros((self.len,5))
        feats[:,0],feats[:,1],feats[:,2],feats[:,3],feats[:,4] = p,r,client_count,util,wload_size
        self.feats = torch.tensor(feats, dtype=torch.float)

        ### our workload mixes, now including individual job type weights
        weight_cols = sorted(
            [column for column in csv.columns if column.startswith('Weight_') and column != 'Weight_Sample_ID'],
            key=lambda column: int(column.split('_')[-1]),
        )
        if not weight_cols:
            raise ValueError('No Weight_0, Weight_1, ... columns were found.')

        weights = csv[weight_cols].to_numpy(dtype=float)
        wl_mix = [
            self.read_workload_str(item, weight_row)
            for item,weight_row in zip(csv['workload_mix'].values,weights)
        ]

        if use_norm_wlmix:
            # The CONDOR paper normalizes workload features by empirical averages.
            # Recompute those averages for the new FlexDC input distribution.
            norm_weights = np.concatenate(wl_mix, axis=0).mean(axis=0)
            norm_weights[-1] = 1.0 # preserve workload weights as probabilities, as in released code
            norm_weights[norm_weights == 0] = 1.0
            self.workload_norm_weights = norm_weights
            wl_mix = [torch.tensor(item/norm_weights, dtype=torch.float) for item in wl_mix]
        else:
            self.workload_norm_weights = np.ones(7, dtype=float)
            wl_mix = [torch.tensor(item, dtype=torch.float) for item in wl_mix]

        if pad_wlmix:
            wl_mix = pad_sequence(wl_mix,batch_first=True,padding_value=0).float()

        #mask = (wl_mix != -1.0)
        #wl_mix_masked = masked_tensor(wl_mix,mask)
        #self.wl_mix = wl_mix_masked
        self.wl_mix = wl_mix
        #self.mask = mask

    def read_workload_str(self,text,weights):
        job_features = np.asarray(json.loads(text), dtype=float)
        weights = np.asarray(weights, dtype=float)
        if job_features.ndim != 2 or job_features.shape[1] != 6:
            raise ValueError(
                'Expected FlexDC workload_mix rows to have six values: '
                '[pmin, pmax, Tmin, Tmax, qos, nodes].'
            )
        if job_features.shape[0] != weights.size:
            raise ValueError(
                f'Workload has {job_features.shape[0]} job types but {weights.size} weights.'
            )
        return np.column_stack((job_features,weights))

    def get_statistics(self):
        print('*** Datacenter Dataset Statistics ***')
        print('** CONDOR Cost Statistics ** ')
        print('Average Cost Power:',torch.mean(self.costs[:,0]).item())
        print('Average Cost Error:',torch.mean(self.costs[:,1]).item())
        print('Average Cost QoS:',torch.mean(self.costs[:,2]).item())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.feats[idx], self.wl_mix[idx],self.costs[idx]


def train_model(model,
                epochs=50,
                lr=1e-3,
                batch_size = 128,
                verbose=False,
                cross_validate = False,
                give_mape=True,
                data_file_path = 'traditional_iso16_fullpilot_AQA_combined_grid_search_results.csv',
                wandb_run=None):

    dc_dataset = DCDataset(data_file_path=data_file_path)
    start_time = time.time()
    # Handles 70-30 CV splitting if desired
    if cross_validate:
        gen = torch.Generator().manual_seed(0)  # this will ensure the same split every time
        train_dc_ds,test_dc_dataset = random_split(dc_dataset, [0.7,0.3], generator=gen)
        dc_train_dataloader = DataLoader(train_dc_ds, batch_size=batch_size, shuffle=True)
        dc_test_dataloader = DataLoader(test_dc_dataset, batch_size=batch_size, shuffle=False)
    else:
        dc_train_dataloader = DataLoader(dc_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model.to(device)

    # Sets up loss recording
    train_epoch_loss_record = []
    test_epoch_loss_record = []

    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_test_loss = 0
        for feat_batch,wl_batch,cost_batch in dc_train_dataloader:
            # puts on device
            feat_batch,wl_batch,cost_batch = feat_batch.to(device),wl_batch.to(device),cost_batch.to(device)
            # Runs our sample through our network
            output = model.forward(feat_batch,wl_batch)

            # Calculates the loss
            loss = criterion(output, cost_batch)

            # Resets the gradient of the optimizer
            optimizer.zero_grad()

            # Performs the backwards pass, finding dL/dW
            loss.backward()
            total_train_loss+=loss.item()

            # Performs one optimizer step
            optimizer.step()
        train_epoch_loss_record.append(total_train_loss/dc_train_dataloader.__len__())

        # Runs through our testing dataset if appropriate
        if cross_validate:
            model.eval()
            with torch.no_grad():
                for feat_batch,wl_batch,cost_batch in dc_test_dataloader:
                    # puts on device
                    feat_batch,wl_batch,cost_batch = feat_batch.to(device),wl_batch.to(device),cost_batch.to(device)
                    # Runs our sample through our network
                    output = model.forward(feat_batch,wl_batch)
                    # Calculates the loss
                    loss = criterion(output, cost_batch)
                    total_test_loss+=loss.item()
            test_epoch_loss_record.append(total_test_loss/dc_test_dataloader.__len__())
        if verbose:
            if cross_validate:
                print('Epoch', epoch, 'Train Loss:' , train_epoch_loss_record[-1], 'Test Loss:', test_epoch_loss_record[-1])
            else:
                print('Epoch', epoch, 'Loss:' , train_epoch_loss_record[-1])

        # Optional W&B logging. Training is unchanged when wandb_run is None.
        if wandb_run is not None:
            metrics = {
                'train_loss': train_epoch_loss_record[-1],
            }
            if cross_validate:
                metrics['heldout_loss'] = test_epoch_loss_record[-1]
            wandb_run.log(metrics, step=epoch)

    running_time = time.time() - start_time
    if verbose:
        time_to_carbon(running_time)
    return model,train_epoch_loss_record,test_epoch_loss_record


def get_workload_mix_features_dict(dict_location = '../data/python_data/'):
    # Legacy CONDOR helper; retained unchanged for comparison with released code.
    workload_filehandler = open(dict_location + 'workload_mixes.dict', 'rb')
    workload_dict = pickle.load(workload_filehandler)
    jobtype_filehandler = open(dict_location + 'job_dictionary.dict', 'rb')
    jobtype_dict = pickle.load(jobtype_filehandler)
    # for normalization
    norm_weights = np.array([252.60714286, 346.34821429, 252.93928571, 297.96428571, 1.26071429,2.71428571])
    # makes new dictionary
    work_feature_dict = {}
    for workload_mix in list(workload_dict.keys()):
        feature_list = []
        for jobtype in workload_dict[workload_mix]:
            # does normalization, calculated kinda manually
            feature_list.append(jobtype_dict[jobtype]/norm_weights)

        work_feature_dict.update({workload_mix:np.array(feature_list)})
    return work_feature_dict


def evaluate_model(model,
                   cross_validate = True,
                   data_file_path = 'traditional_iso16_fullpilot_AQA_combined_grid_search_results.csv'):
    batch_size = 128
    dc_dataset = DCDataset(data_file_path=data_file_path)

    # Handles 70-30 CV splitting if desired
    if cross_validate:
        gen = torch.Generator().manual_seed(0)  # this will ensure the same split every time
        train_dc_ds,test_dc_dataset = random_split(dc_dataset, [0.7,0.3], generator=gen)
        dc_train_dataloader = DataLoader(train_dc_ds, batch_size=batch_size, shuffle=False)
        dc_test_dataloader = DataLoader(test_dc_dataset, batch_size=batch_size, shuffle=False)
    else:
        dc_train_dataloader = DataLoader(dc_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    y_true_train = []
    y_pred_train = []
    y_true_test = []
    y_pred_test = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for feat_batch,wl_batch,cost_batch in dc_train_dataloader:
            # puts on device
            feat_batch,wl_batch,cost_batch = feat_batch.to(device),wl_batch.to(device),cost_batch.to(device)
            # Runs our sample through our network
            output = model.forward(feat_batch,wl_batch)
            # records
            for true_cost in cost_batch:
                y_true_train.append(true_cost.detach().cpu().numpy())
            for pred_cost in output:
                y_pred_train.append(pred_cost.detach().cpu().numpy())
        # Runs through our testing dataset if appropriate
        if cross_validate:
            for feat_batch,wl_batch,cost_batch in dc_test_dataloader:
                # puts on device
                feat_batch,wl_batch,cost_batch = feat_batch.to(device),wl_batch.to(device),cost_batch.to(device)
                # Runs our sample through our network
                output = model.forward(feat_batch,wl_batch)
                for true_cost in cost_batch:
                    y_true_test.append(true_cost.detach().cpu().numpy())
                for pred_cost in output:
                    y_pred_test.append(pred_cost.detach().cpu().numpy())

    y_true_train = np.array(y_true_train)
    y_pred_train = np.array(y_pred_train)
    y_true_test = np.array(y_true_test)
    y_pred_test = np.array(y_pred_test)

    if cross_validate:
        print('MSE  ||','Train:',str(mean_squared_error(y_true_train,y_pred_train)),'| Test:',str(mean_squared_error(y_true_test,y_pred_test)))
        print('MAPE ||','Train:',str(mean_absolute_percentage_error(y_true_train,y_pred_train)),'| Test:',str(mean_absolute_percentage_error(y_true_test,y_pred_test)))
        print('Total MSE  ||','Train:',str(mean_squared_error(y_true_train.sum(axis=1),y_pred_train.sum(axis=1))),'| Test:',str(mean_squared_error(y_true_test.sum(axis=1),y_pred_test.sum(axis=1))))
        print('Total MAPE ||','Train:',str(mean_absolute_percentage_error(y_true_train.sum(axis=1),y_pred_train.sum(axis=1))),'| Test:',str(mean_absolute_percentage_error(y_true_test.sum(axis=1),y_pred_test.sum(axis=1))))
    else:
        print('MSE  ||','Train:',str(mean_squared_error(y_true_train,y_pred_train)))
        print('MAPE ||','Train:',str(mean_absolute_percentage_error(y_true_train,y_pred_train)))
        print('Total MSE  ||','Train:',str(mean_squared_error(y_true_train.sum(axis=1),y_pred_train.sum(axis=1))))
        print('Total MAPE ||','Train:',str(mean_absolute_percentage_error(y_true_train.sum(axis=1),y_pred_train.sum(axis=1))))
    return y_true_train,y_pred_train,y_true_test,y_pred_test


def time_to_carbon(training_time,
                   compute_watts = 330):
    # Because we submitted the revised paper to HotCarbon 24',
    # we thought it would be prudent to include carbon cost statistics
    # about our model's training time and shit like that.
    # This method is just so I can write this down once and not have to do it
    # over and over again.

    # Data taken from this website: https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator#results
    # my dimensional analysis may be cooked. sorry

    # NOTE - training time should be in seconds.
    # NOTE - compute watts assumes my personal laptop.
    metric_tons_co2 = compute_watts*2.7778e-7*.0007 * training_time
    print('Training Time:', training_time)
    print('Metric Tons of Co2:',metric_tons_co2)
