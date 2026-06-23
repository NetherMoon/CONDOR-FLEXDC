import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *
# The Set Transformer, from https://github.com/juho-lee/set_transformer/tree/master.
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
        x = self.enc(X)
        x = self.dec(x)
        return x
    
class DataCenterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        gpu = torch.device('cuda')
        self.device = gpu
        self.activation = nn.SiLU()
        self.linear1 = nn.Linear(8,64)
        self.linear2 = nn.Linear(64,1)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x

def model_pr_descent(p_init,r_init, server_features, model, iterations=100, lr=0.01):
    P_record = []
    R_record = []
    cost_record = []
    input = torch.tensor([p_init,r_init] + list(server_features)).float() # TODO - add server features
    input = torch.autograd.Variable(input, requires_grad=True)
    model.zero_grad()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input.to(device)
    model.to(device)
    for _ in range(iterations):
        output = model(input)
        output.backward()
        grad = input.grad
        with torch.no_grad():
            input[0] -= lr * grad[0] # P Update
            input[1] -= lr * grad[1] # R Update
        # stores. why not 
        P_record.append(input[0].item())
        R_record.append(input[1].item())
        cost_record.append(output.item())
    #return input[0].item(),input[1].item() # returns predicted P, R
    return P_record, R_record,cost_record

def train_model(model,X,Y,epochs=50,lr=1e-3):
    # TODO - update this with proper batching or dataloaders. whatever
    X = torch.tensor(X.astype(float)).float()
    Y = torch.tensor(Y.astype(float)).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    gpu = torch.device('cuda')
    loss_record = []
    batch_size = 32
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for index in range(len(X)):
            # Grabs our sample from our training data 
            input_x = X[index]
            input_y = Y[index].unsqueeze(0)
            # Runs our sample through our network
            output = model.forward(input_x)
            # Calculates the loss
            loss = criterion(output, input_y)
            # Resets the gradient of the optimizer
            optimizer.zero_grad()
            # Performs the backwards pass, finding dL/dW
            loss.backward()
            total_loss+=loss.item()
            # Performs one optimizer step
            optimizer.step()
        loss_record.append(total_loss/len(Y))
        total_loss = 0
    return model,loss_record
