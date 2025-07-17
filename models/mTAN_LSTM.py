"""

Benchmark LSTM model for prediction in time using data across several watersheds 

Author: Abhinav Gupta (Created: 2 May 2022)

"""
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from random import choices
import copy
import random
import gc
import math
from numpy.random import RandomState

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
device = "cuda" if torch.cuda.is_available() else "cpu"

# function to compute NSE
def computeNSE(obs, pred):
    sse = np.sum((obs - pred)**2)
    sst = np.sum((obs - np.mean(obs))**2)
    nse = 1 - sse/sst
    return nse

########################################################################################################
# LSTM classes
########################################################################################################
# build a dataset class
class CustomData(Dataset):
    def __init__(self, data, mean_X, std_X, stdy_list, data_inds, seq_len):
        super(Dataset, self).__init__()
        self.data = data
        self.L = seq_len            # sequence length
        self.mx = mean_X            # mean of predictor variables
        self.sdx = std_X            # standard deviation of predictor variables
        self.stdy_list = stdy_list  # maximum values of SSC for each COMID_ID
        self.index_map = data_inds  # indices to be used for model run (to exclude the indices related to NaN)
       
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        comid_ind, tind = idx
        x1 = self.data[comid_ind][tind - self.L+1 : tind+1, 1:]
        x1 = torch.div(x1 - self.mx, self.sdx)
        y1 = torch.div(self.data[comid_ind][tind,0], self.stdy_list[comid_ind])
        mask1 = torch.ones(x1.shape)
        mask1[torch.isnan(x1)] = 0.0
        tsteps1 = torch.arange(1,self.L+1)                                     ################################### should it be reversed????????????
        return comid_ind, x1, y1, self.stdy_list[comid_ind], tsteps1, mask1

# custom loss function
class CustomLoss(nn.Module):
    def __init__(self, q):
        super(CustomLoss, self).__init__()
        self.q = q
        
    def forward(self, output, target):
        res = output-target
        loss = (1-self.q)*torch.sum(res[res>0]) - self.q*torch.sum(res[res<=0])
        return loss

# Custom sampler function 
class SubsetSampler(Sampler):
    def __init__(self, indices, generator=None):
        super(Sampler, self).__init__()
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        for i in self.indices:
            yield i

    def __len__(self) -> int:
        return len(self.indices)

# function to collate data
def CustomCollate_fn(batch):
    D = batch[0][1].shape[1]

    combined_tt, inverse_indices = torch.unique(torch.cat([ex[4] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    combined_comind = torch.zeros([len(batch)]).to(device)
    combined_x = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_y = torch.zeros([len(batch)]).to(device)
    combined_stdy = torch.zeros([len(batch)]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    
    for b, (comid_ind, x, y, stdy, tt, mask) in enumerate(batch):
        tt = tt.to(device) 
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)
        stdy = stdy.to(device)

        combined_comind[b] = comid_ind 
        combined_x[b] = x
        combined_y[b] = y
        combined_mask[b] = mask
        combined_stdy[b] = stdy

    combined_tt = combined_tt.repeat(len(batch),1).float()

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)
    combined_x[torch.isnan(combined_x)] = 0.0

    data_dict = {
        "comid_ind": combined_comind,
        "data": combined_x, 
        "time_steps": combined_tt,
        "mask": combined_mask,
        "y": combined_y,
        "stdy": combined_stdy}

    return data_dict

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1, device='cuda', non_linear_transform=False):       
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads 
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)]).to(device)
        self.non_linear_transform = non_linear_transform
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)           #######################   What does transpose(-2,-1) achieve?
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)    # same score is used along each input dimension
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]          
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        if self.non_linear_transform:
            x = self.linears[-1](x)                  
        return x
    
class mtan(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, device='cuda', non_linear_transform=False):       
        super(mtan, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(input_dim, nhidden, embed_time, num_heads, device, non_linear_transform)

        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2)).to(device)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1).to(device)
            self.linear = nn.Linear(1, 1).to(device)
        
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, x, mask, time_steps):           ######## what is the format of x and timesteps??
        time_steps = time_steps.cpu()
        #mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, x, mask)
        #out, _ = self.gru_rnn(out)
        #out = self.hiddens_to_z0(out)
        return out
    
# define LSTM model class
class mTANLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_lstm, n_lstm_layers, hidden_dim_mtan, embed_dim_mtan, num_heads, 
                 Lseq, output_dim, non_linear_transform, RS_indices):
        super(mTANLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_lstm = hidden_dim_lstm
        self.n_lstm_layers = n_lstm_layers
        self.non_linear_transform = non_linear_transform
        #self.tan = mtan(input_dim, torch.arange(0,Lseq)/(Lseq-1), latent_dim=2, nhidden=hidden_dim_mtan, embed_time=embed_dim_mtan, 
        #                num_heads=num_heads, learn_emb=True, non_linear_transform=non_linear_transform)
        self.tan = mtan(len(RS_indices), torch.arange(0,Lseq)/(Lseq-1), 
                        latent_dim=2, nhidden=len(RS_indices), embed_time=embed_dim_mtan, 
                        num_heads=num_heads, learn_emb=True, 
                        non_linear_transform=non_linear_transform)
        if non_linear_transform:
            self.lstm = nn.LSTM(hidden_dim_mtan, hidden_dim_lstm, n_lstm_layers, batch_first = True)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim_lstm, n_lstm_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim_lstm, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.40)
        self.RS_inds = RS_indices

    def forward(self, vals, time_steps, mask):
        #x = self.tan(vals, mask, time_steps)
        #x = x + vals
        x1 = self.tan(vals[:,:,self.RS_inds], mask[:,:,self.RS_inds], time_steps) 
        x2 = torch.cat(
            (torch.zeros(x1.shape[0],x1.shape[1],np.min(self.RS_inds)).to(device),
            x1, 
             torch.zeros(x1.shape[0],x1.shape[1],input_dim-np.max(self.RS_inds)-1).to(device)
             ), 
            axis=2)
        x = vals + x2

        h0 = torch.zeros(self.n_lstm_layers, x.size(0), self.hidden_dim_lstm, device = x.device).requires_grad_()
        c0 = torch.zeros(self.n_lstm_layers, x.size(0), self.hidden_dim_lstm, device = x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0, c0))
        #out = self.fc(out[:,-1,:]) #use hn instead, use relu before fc
        out = self.fc(self.dropout(hn[0,:,:]))
        #out = self.fc(hn[0,:,:])
        #out=self.fc1(out)
        return out

# define the module to train the model
def train_mod(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    tr_loss  = 0
    for batch_no, batch in enumerate(dataloader):
        #X, y = X.to(device), y.to(device)

        comid_inds, X, tp, mask, y, stdy = batch['comid_ind'], batch['data'], batch['time_steps'], batch['mask'], batch['y'], batch['stdy']

        # Compute prediction error
        pred = model(X, tp, mask)
        loss = loss_fn(pred, y.view(len(y),1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()

        del loss, pred

    tr_loss /= size
    return tr_loss, model.state_dict()

# define the module to test the model
def test_mod(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    sse = 0
    ynse, pred_list, COMID_inds, stdy_out = [], [], [], []
    with torch.no_grad():
        for batch_no, batch in enumerate(dataloader):
        #X, y = X.to(device), y.to(device)

            comid_inds, X, tp, mask, y, stdy = batch['comid_ind'], batch['data'], batch['time_steps'], batch['mask'], batch['y'], batch['stdy']
            y = y.view(len(y),1)
            pred = model(X, tp, mask)
            
            sse += torch.sum((pred - y)**2)
            ynse.append(y)
            pred_list.append(pred)
            COMID_inds.append(comid_inds)
            stdy_out.append(stdy)

    ynse = torch.cat(ynse)
    pred_list = torch.cat(pred_list)
    COMID_inds = torch.cat(COMID_inds)
    stdy_out = torch.cat(stdy_out)
    sst = torch.sum((ynse - torch.mean(ynse))**2)
    nse = 1 - sse/sst

    print(f"NSE: {nse.item():>8f} \n")
    return nse.item(), pred_list, ynse, COMID_inds, stdy_out