import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import random
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
    
class DataCenterModel(torch.nn.Module):
    def __init__(self, 
                 dim_job_mix = 7,
                 dim_dc_features = 5,
                 # Below arguments about Set Transformer parameter
                 st_num_outputs=1, 
                 st_dim_output=1024-5,
                 st_num_inds=32, 
                 st_dim_hidden=512, 
                 st_num_heads=4, 
                 st_ln=False,
                # Below arguments about linear layer things
                 linear_dim_hidden = 512,
                 skip_connections = True
                 ):
        super().__init__()

        ### Note = 7 inputs are for 7 workload features - those are
        # min_power	
        # max_power	
        # min_time	
        # max_time	
        # qos	
        # nodes
        # weight

        ### Note - tabular terms added are 
        # P
        # R
        # client count (number of servers)
        # utilization
        # number of jobs in workload mix
        self.skip_connections = skip_connections
        gpu = torch.device('cuda')
        self.device = gpu
        self.activation = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        # Linear layers used in skip
        self.linear1 = nn.Linear(st_dim_output+dim_dc_features,linear_dim_hidden)
        self.linear2 = nn.Linear(linear_dim_hidden,linear_dim_hidden)
        self.linear3 = nn.Linear(linear_dim_hidden,linear_dim_hidden)
        # Linear output heads for each cost thing
        self.cost_output = nn.Linear(linear_dim_hidden,3)
        # Set Transformer blocks 
        self.enc_long = nn.Sequential(
                ISAB(dim_job_mix, st_dim_hidden, st_num_heads, st_num_inds, ln=st_ln),
                ISAB(st_dim_hidden, st_dim_hidden, st_num_heads, st_num_inds, ln=st_ln))
        self.dec_long = nn.Sequential(
                PMA(st_dim_hidden, st_num_heads, st_num_outputs, ln=st_ln),
                SAB(st_dim_hidden, st_dim_hidden, st_num_heads, ln=st_ln),
                SAB(st_dim_hidden, st_dim_hidden, st_num_heads, ln=st_ln),
                nn.Linear(st_dim_hidden, st_dim_output))
        self.enc_short = nn.Sequential(
            SAB(dim_in=dim_job_mix, dim_out=st_dim_hidden, num_heads=st_num_heads),
            SAB(dim_in=st_dim_hidden, dim_out=st_dim_hidden, num_heads=st_num_heads),
        )
        self.dec_short = nn.Sequential(
            PMA(st_dim_hidden, st_num_heads, st_num_outputs),
            nn.Linear(in_features=st_dim_hidden, out_features=st_dim_output),
        )

    def forward(self, sim_features, workload_mix):
        # TODO - experiment w/residuals vs multilayer fusion 
        # Runs the workload through the Set Trnasfomer encoder/decoder 
        # and single linear layer
        workload_feature = self.dec_short(self.enc_short(workload_mix)).squeeze() # 128
        # concatenates workload with sim_config,p,r, runs through two layer MLP
        # right now, sim_config is just util
        
        x1 = torch.cat((workload_feature,sim_features),dim=-1) # st_out + 5
        x2 = self.activation(self.linear1(x1))    # st_out+5 -> lin_hidden
        x3 = self.activation(self.linear2(x2)) # lin_hidden -> lin_hidden
        if self.skip_connections:
            x4 = self.activation(self.linear3(x3+x2)) # lin_hidden -> lin_hidden
            out = self.cost_output(x4+x3)             # lin_hidden -> out_dim
        else:
            x4 = self.activation(self.linear3(x3)) # lin_hidden -> lin_hidden
            out = self.cost_output(x4)             # lin_hidden -> out_dim
        return out


# Modules from https://github.com/juho-lee/set_transformer/tree/master.
# Left unmodified, but used in our sim-model.
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# stuff from the Set Transformer paper, from https://github.com/juho-lee/set_transformer/tree/master.
# Left unmodified but unused, as reference. 
class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))
    
class SmallDeepSet(nn.Module):
    def __init__(self, pool="max"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(x)
        return x


class SmallSetTransformer(nn.Module):
    def __init__(self,):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=1, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)