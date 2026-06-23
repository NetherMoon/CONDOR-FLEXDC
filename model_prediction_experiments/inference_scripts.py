import torch
from torch.nn import Softmax
import numpy as np
import pickle

def model_pr_descent(p_init,
                     r_init, 
                     workload_name, # Name of workload
                     model,
                     cost_weights = [0.1,1,1], #TODO find better weights
                     client_count = 500, # for now, should be client count, util
                     util = 0.8,
                     dict_location = '../data/', 
                     iterations=150, 
                     lr=1e-2):
    #TODO - fix to work with new data and dataloader 
    P_record,R_record, W_record, P_grad_record,R_grad_record,W_grad_record = [], [], [], [], [], []

    # Loads WL Mix, adds initial uniform weights TODO - add capability for different initialization
    work_feature_dict = get_workload_mix_features_dict(dict_location)
    workload_mix = work_feature_dict[workload_name] # work_mix_size x 6 array 
    workload_mix_size = len(workload_mix)
    workload_weights = np.ones(workload_mix_size)/workload_mix_size # uniform start 
    workload_mix_combined = np.concatenate((workload_mix,np.array([workload_weights]).T),axis=1)
    workload = torch.autograd.Variable(torch.tensor(workload_mix_combined), requires_grad=True)

    # Adds Makes sim config feature vector

    cost_record = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    sim_config =  torch.autograd.Variable(torch.tensor([p_init,r_init,client_count,util,workload_mix_size]), requires_grad=True)

    model.zero_grad()
    model.eval()
    model.to(device)
    # drops everything on device

    #sim_config = sim_config.unsqueeze(0).to(device)
    sim_config = sim_config.to(device)
    sim_config.retain_grad()
    workload = workload.unsqueeze(0).float().to(device)
    workload.retain_grad()

    P_record.append(p_init)
    R_record.append(r_init)
    W_record.append(workload[0][:,6].cpu().detach().numpy())
    
    # gradient descent step proper 
    for iter in range(iterations):
        # Runs model
        c_power,c_error,c_qos = model(sim_config,workload)

        # Weights Cost TODO 
        cost = cost_weights[0]*c_power + cost_weights[1]*c_error + cost_weights[2]*c_qos
        #print(type(cost))
        # Takes cost gradient, finds P/R/W gradients
        cost.retain_grad()
        cost.backward()
        sim_config_grad = sim_config.grad
        #print(type(sim_config_grad))
        #print(sim_config_grad.shape)
        p_step = sim_config_grad[0]
        r_step = sim_config_grad[1]
        workload_grad = workload.grad
        weight_step = workload_grad[0][:,6]

        # Gradient Descent step 
        with torch.no_grad():
            sim_config[0]  -= lr * p_step # P Update
            sim_config[1]  -= lr * r_step # R Update

            # Weight step - uses Projected GD because we 
            # need to keep W in the feasability set of the 
            # R^|workload mix| unit ball 
            # Because it's more involved I write things out more so its more easy to follow
            w_current = workload[0][:,6]
            softmax = Softmax(dim=0)
            update = softmax(w_current - weight_step) - w_current
            workload[0][:,6] += lr * update

        # Stores all updated values, gradients, and costs for reference 
        P_record.append(sim_config[0].item())
        R_record.append(sim_config[1].item())
        W_record.append(workload[0][:,6].cpu().detach().numpy())
        P_grad_record.append(p_step.item())
        R_grad_record.append(r_step.item())
        W_grad_record.append(weight_step.detach().cpu().numpy())
        cost_record.append(cost.item())
        # Zeroes the gradients - very important!
        sim_config_grad.zero_()
        workload_grad.zero_()
    # finds cost of last iterate
    c_power,c_error,c_qos = model(sim_config,workload)
    cost = cost_weights[0]*c_power + cost_weights[1]*c_error + cost_weights[2]*c_qos
    cost_record.append(cost.item())
    #return input[0].item(),input[1].item() # returns predicted P, R
    return np.array(P_record), np.array(R_record),np.array(W_record),np.array(P_grad_record), np.array(R_grad_record),np.array(W_grad_record),np.array(cost_record)

def save_model_output(p_final,
                      r_final,
                      w_final,
                      experiment_name,
                      profile_config = '../configs/q_grid_search/scc-applications-20200108.ini',
                      exp_config = '\"../configs/q_grid_search/W5.ini\"',
                      num_clients = 500,
                      weight_file_dir = 'hotcarbon_24_weights'
                      ):
    # This method is to make it easier to turn model run results into a simulator
    # configuration. It prints the sim command you'll need to run and save the 
    # weights to the appropriate file. 

    # Example of weight file name: optWeights_hotcarbon_exp_w3.csv
    # Example of sim running command: 
    # python -u master.py -p GPS -x paper_w5_real -g -b 1.007595 -r 0.582944 -u -z 0.000000 --profileConfig 
    # ../configs/q_grid_search/scc-applications-20200108.ini 
    # --experimentConfig "../configs/q_grid_search/W5.ini" --simulation 500   
    # -o _pap_results_w5 --perf-variation=0.00 --phase-length=10800 --random-seed=20

    # Prints weights to file
    weights_name = weight_file_dir + '/' + 'optWeights_' + experiment_name + '.csv'
    np.savetxt(weights_name, [w_final], delimiter=',',fmt='%f')

    # Makes Terminal Command
    exp_command_name = ''
    exp_command_name += 'python -u master.py -p GPS -x ' + experiment_name # boilerplate stuff + experiment name
    exp_command_name += ' -g -b ' + str(p_final) + ' -r ' + str(r_final) # P and R
    exp_command_name += ' -u -z 0.000000 ' # no clue tbh
    exp_command_name += ' --profileConfig ' + profile_config # profile configuration
    exp_command_name += ' --experimentConfig ' + exp_config # experiment configuration
    exp_command_name += ' --simulation ' + str(num_clients) # clients
    exp_command_name += ' -o '  + '_' + experiment_name # output file
    exp_command_name += ' --perf-variation=0.00 --phase-length=10800 --random-seed=20' # other random stuff
    print(exp_command_name)




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