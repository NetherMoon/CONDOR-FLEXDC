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
    # Updated! TODO implement optinal things
    def __init__(self, 
                 data_file_path = '../data/all_data.csv',
                 use_norm_pr = True, # uses default normalized P and R
                 use_norm_cost = True, # Normalizes power and error cost by number of clients, and QOS by number of jobs
                 use_norm_wlmix = True, # HIGHLY reccomend set this to true
                 pad_wlmix = True
                 ):  
        csv = pd.read_csv(data_file_path)
        self.len = csv.shape[0]

        ### Cost - our labels!
        c_power,c_error,c_qos = np.array(list(csv['cost_power'].values)),np.array(list(csv['cost_error'].values)),np.array(list(csv['cost_qos']))
        if use_norm_cost:
            server_counts = np.array(list(csv['client_count'].values))
            job_mix_length = np.array(list(csv['workload_mix_size']))
            c_power,c_error,c_qos = (c_power*120)/server_counts,(c_error*200)/server_counts,c_qos/job_mix_length
        costs = np.zeros((self.len,3))
        costs[:,0],costs[:,1],costs[:,2] = c_power,c_error,c_qos
        self.costs = torch.tensor(costs, dtype=torch.float)

        ### Our tabular server features
        if use_norm_pr:
            p,r = np.array(list(csv['p_norm'].values)),np.array(list(csv['r_norm'].values))
        else:
            # Even when not normalized, the range is still a 
            # little high for a NN model - so we divide by a thousand 
            # so we are predicting kilowatts, which is convenient.
            p,r = np.array(list(csv['p'].values))/1000,np.array(list(csv['r'].values))/1000
        client_count,util,wload_size = list(csv['client_count'].values),list(csv['util'].values),list(csv['workload_mix_size'])
        feats = np.zeros((self.len,5))
        feats[:,0],feats[:,1],feats[:,2],feats[:,3],feats[:,4] = p,r,client_count,util,wload_size
        self.feats = torch.tensor(feats, dtype=torch.float)

        ### our workload mixes, now including indivivudal job type weights
        wl_mix = list(csv['workload_mix'].values)
        if use_norm_wlmix:
            norm_weights = np.array([252.60714286, 346.34821429, 252.93928571, 297.96428571, 1.26071429,2.71428571,1])
            wl_mix = [torch.tensor(self.read_workload_str(item)/norm_weights) for item in wl_mix]
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
        n_jobs = str.count('[]')-1
        str = str.replace('[','').replace(']','').replace('\n','')
        thing = np.fromstring(str,dtype=float, sep=' ')
        return np.reshape(thing,(n_jobs,7))
    def get_statistics(self):
        print('*** Datacenter Dataset Statistics ***')
        print('** Cost Statistics ** ')
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
                give_mape=True): # handles if you want given evaluation metric to be MSE or MAPE (MAPE for paper)
    
    dc_dataset = DCDataset()
    start_time = time.time()
    # Handles 70-30 CV splitting if desired
    if cross_validate:
        gen = torch.Generator().manual_seed(0)  # this will ensure the same split every time
        train_dc_ds,test_dc_dataset = random_split(dc_dataset, [0.7,0.3], generator=gen)
        dc_train_dataloader =   DataLoader(train_dc_ds, batch_size=batch_size, shuffle=True)
        dc_test_dataloader =    DataLoader(test_dc_dataset,  batch_size=batch_size, shuffle=True)
    else:
        dc_train_dataloader = DataLoader(dc_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Sets up loss recording
    train_epoch_loss_record = []
    test_epoch_loss_record = []

    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
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
        total_train_loss = 0
        total_test_loss = 0
    running_time = time.time() - start_time
    if verbose:
        time_to_carbon(running_time)
    return model,train_epoch_loss_record,test_epoch_loss_record

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

def evaluate_model(model,cross_validate = True):
    # TODO fix. this was using deprecated model and stuff it won't work now 
    # For now, this does nothing in batch

    # X input should be iterable with each sample being the following:
    # [Workload Name,any simulation configuration things, P, R]
    batch_size = 128
    dc_dataset = DCDataset()

    # Handles 70-30 CV splitting if desired
    if cross_validate:
        gen = torch.Generator().manual_seed(0)  # this will ensure the same split every time
        train_dc_ds,test_dc_dataset = random_split(dc_dataset, [0.7,0.3], generator=gen)
        dc_train_dataloader =   DataLoader(train_dc_ds, batch_size=batch_size, shuffle=True)
        dc_test_dataloader =    DataLoader(test_dc_dataset,  batch_size=batch_size, shuffle=True)
    else:
        dc_train_dataloader = DataLoader(dc_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    y_true_train = []
    y_pred_train = []
    y_true_test = []
    y_pred_test = []
    model.to(device)
    for feat_batch,wl_batch,cost_batch in dc_train_dataloader:
            # puts on device
            feat_batch,wl_batch,cost_batch = feat_batch.to(device),wl_batch.to(device),cost_batch.to(device)
            # Runs our sample through our network
            output = model.forward(feat_batch,wl_batch)
            # records
            for true_cost in cost_batch:
                y_true_train.append(true_cost.numpy(force=True))
            for pred_cost in output:
                y_pred_train.append(pred_cost.numpy(force=True))
    # Runs through our testing dataset if appropriate 
    if cross_validate:
        for feat_batch,wl_batch,cost_batch in dc_test_dataloader:
            # puts on device
            feat_batch,wl_batch,cost_batch = feat_batch.to(device),wl_batch.to(device),cost_batch.to(device)
            # Runs our sample through our network
            output = model.forward(feat_batch,wl_batch)
            for true_cost in cost_batch:
                y_true_test.append(true_cost.numpy(force=True))
            for pred_cost in output:
                y_pred_test.append(pred_cost.numpy(force=True))
    if cross_validate:
        print('MSE  ||','Train:',str(mean_squared_error(y_true_train,y_pred_train)),'| Test:',str(mean_squared_error(y_true_test,y_pred_test)))
        print('MAPE ||','Train:',str(mean_absolute_percentage_error(y_true_train,y_pred_train)),'| Test:',str(mean_absolute_percentage_error(y_true_test,y_pred_test)))
    else:
        print('MSE  ||','Train:',str(mean_squared_error(y_true_train,y_pred_train)))
        print('MAPE ||','Train:',str(mean_absolute_percentage_error(y_true_train,y_pred_train)))
    return np.array(y_true_train),np.array(y_pred_train),np.array(y_true_test),np.array(y_pred_test)
    
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