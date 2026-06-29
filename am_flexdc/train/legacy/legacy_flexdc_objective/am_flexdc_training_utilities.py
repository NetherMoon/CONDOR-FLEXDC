import ast
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


class DCDataset(Dataset):
    # FlexDC adaptation of the original CONDOR dataset loader.
    def __init__(self, 
                 data_file_path = '../data/flexdc_all_data.csv',
                 use_norm_pr = False, # False: use physical Pbar/R in kW per server
                 use_norm_cost = False, # Original CONDOR cost normalizers do not apply to FlexDC terms
                 use_norm_wlmix = True, # Converts FlexDC workload power to kW and time to hours
                 pad_wlmix = True
                 ):  
        csv = pd.read_csv(data_file_path)
        self.len = csv.shape[0]

        ### Cost - our labels!
        # FlexDC Eq. (10): logged labels are M, psi*Ctrack, and beta*CQoS,
        # so C_total = M_RSR + C_track + C_Qos without applying beta/psi again.
        c_mrsr = np.array(list(csv['M_RSR'].values))
        c_track = np.array(list(csv['C_track'].values))
        c_qos = np.array(list(csv['C_Qos'].values))

        # Verify that the converted labels still reconstruct the logged full FlexDC objective.
        if not np.allclose(
            np.array(list(csv['C_total'].values)),
            c_mrsr + c_track + c_qos,
            rtol=1e-8,
            atol=1e-8,
        ):
            raise ValueError('C_total must equal M_RSR + C_track + C_Qos for every row.')

        if use_norm_cost:
            raise ValueError(
                'The original CONDOR 120/200/job-count cost normalizers are not valid '
                'for the FlexDC objective terms. Keep use_norm_cost=False.'
            )

        costs = np.zeros((self.len,3))
        costs[:,0],costs[:,1],costs[:,2] = c_mrsr,c_track,c_qos
        self.costs = torch.tensor(costs, dtype=torch.float)

        ### Our tabular server features
        # The CONDOR paper samples Pbar and R in kW/server. The FlexDC CSV stores
        # total watts, so recover the physical kW/server sweep values here.
        server_counts = np.array(list(csv['server_count'].values), dtype=float)
        if use_norm_pr:
            p,r = np.array(list(csv['Pbar_ratio'].values)),np.array(list(csv['R_ratio'].values))
        else:
            p = np.array(list(csv['P_actual'].values), dtype=float)/(1000*server_counts)
            r = np.array(list(csv['R_actual'].values), dtype=float)/(1000*server_counts)
        client_count = list(csv['server_count'].values)
        util = list(csv['utilization'].values)
        wload_size = list(csv['workload_mix_size'].values)
        feats = np.zeros((self.len,5))
        feats[:,0],feats[:,1],feats[:,2],feats[:,3],feats[:,4] = p,r,client_count,util,wload_size
        self.feats = torch.tensor(feats, dtype=torch.float)

        ### our workload mixes, now including individual job type weights
        # Each FlexDC row stores [pmin W, pmax W, Tmin s, Tmax s, qos, nodes, weight].
        # Use physical unit conversions rather than CONDOR's legacy data-derived normalizer:
        # watts -> kW and seconds -> hours. The remaining fields are already small/unitless.
        wl_mix = list(csv['workload_mix'].values)
        if use_norm_wlmix:
            norm_weights = np.array([1000, 1000, 3600, 3600, 1, 1, 1], dtype=float)
            wl_mix = [torch.tensor(self.read_workload_str(item)/norm_weights, dtype=torch.float) for item in wl_mix]
        else:
            wl_mix = [torch.tensor(self.read_workload_str(item), dtype=torch.float) for item in wl_mix]
        if pad_wlmix:
            wl_mix = pad_sequence(wl_mix,batch_first=True,padding_value=0).float()

        #mask = (wl_mix != -1.0)
        #wl_mix_masked = masked_tensor(wl_mix,mask)
        #self.wl_mix = wl_mix_masked
        self.wl_mix = wl_mix
        #self.mask = mask

    def read_workload_str(self,str):
        workload_mix = np.asarray(ast.literal_eval(str), dtype=float)
        if workload_mix.ndim != 2 or workload_mix.shape[1] != 7:
            raise ValueError(
                'Expected workload_mix rows to have seven values: '
                '[pmin, pmax, Tmin, Tmax, qos, nodes, weight].'
            )
        return workload_mix

    def get_statistics(self):
        print('*** Datacenter Dataset Statistics ***')
        print('** FlexDC Objective Component Statistics ** ')
        print('Average M_RSR:',torch.mean(self.costs[:,0]).item())
        print('Average C_track:',torch.mean(self.costs[:,1]).item())
        print('Average C_Qos:',torch.mean(self.costs[:,2]).item())

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
                data_file_path = '../data/flexdc_all_data.csv',
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
    device = torch.device("cpu")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

        # Optional W&B logging. Training behavior is unchanged when wandb_run is None.
        if wandb_run is not None:
            metrics = {
                'epoch': epoch,
                'train_loss': train_epoch_loss_record[-1],
            }
            if cross_validate:
                metrics['heldout_loss'] = test_epoch_loss_record[-1]
            wandb_run.log(metrics)

        total_train_loss = 0
        total_test_loss = 0
    running_time = time.time() - start_time
    if verbose:
        time_to_carbon(running_time)
    return model,train_epoch_loss_record,test_epoch_loss_record


# Legacy CONDOR helper. It is not used by FlexDC training; FlexDC inference will be adapted separately.
def get_workload_mix_features_dict(dict_location = '../data/python_data/'):
    # loads dictionaries
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
                   data_file_path = '../data/flexdc_all_data.csv'):
    # Evaluates the three FlexDC objective components, then reconstructs C_total.
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
    
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
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
        print('C_total MSE  ||','Train:',str(mean_squared_error(y_true_train.sum(axis=1),y_pred_train.sum(axis=1))),'| Test:',str(mean_squared_error(y_true_test.sum(axis=1),y_pred_test.sum(axis=1))))
        print('C_total MAPE ||','Train:',str(mean_absolute_percentage_error(y_true_train.sum(axis=1),y_pred_train.sum(axis=1))),'| Test:',str(mean_absolute_percentage_error(y_true_test.sum(axis=1),y_pred_test.sum(axis=1))))
    else:
        print('MSE  ||','Train:',str(mean_squared_error(y_true_train,y_pred_train)))
        print('MAPE ||','Train:',str(mean_absolute_percentage_error(y_true_train,y_pred_train)))
        print('C_total MSE  ||','Train:',str(mean_squared_error(y_true_train.sum(axis=1),y_pred_train.sum(axis=1))))
        print('C_total MAPE ||','Train:',str(mean_absolute_percentage_error(y_true_train.sum(axis=1),y_pred_train.sum(axis=1))))
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
