import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

def get_GD_file_names(directory='aqa_sim_data/realExperiment'):
    return glob.glob('../' + directory + '/GD*')

def get_sim_hour_steps(file_name):
    temp_data = pd.read_csv(file_name).to_numpy()
    # note - Cost Col 2, P col 6, R col 7, Pcoef col 8, Rcoef col 9
    cost = temp_data[0:,2]
    cost_power = temp_data[0:,3]
    cost_error = temp_data[0:,4]
    cost_qos = temp_data[0:,5]
    P_watts = temp_data[0:,6]
    R_watts = temp_data[0:,7]
    P_coef = temp_data[0:,8]
    R_coef = temp_data[0:,9]
    
    return cost,cost_power,cost_error,cost_qos,P_watts,R_watts,P_coef,R_coef