import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from random import choices
import copy
import random
import gc

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler
from torchvision import datasets
from torchvision.transforms import ToTensor
from models.mTAN_LSTM import computeNSE, mTANLSTMModel, CustomData, SubsetSampler, CustomCollate_fn, train_mod, test_mod
from load_data import readRawData, TrainTestDates, TrainingTestingData, Tesorformatting

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
Lseq = 120       # Sequence length
epochs = 6     # Number of epochs
nseeds = 8      # Number of seeds
hidden_dim_lstm, n_lstm_layers, embed_dim_mtan, num_heads, output_dim = 256, 1, 128, 16, 1       # hidden dimension, number of layers, embedding diemsnion for mTAN, number of heads, output dimension
lrate, optim_lr = 10**(-3), 0.7    # Learning rate, scaling factor of learning rate after each epoch
N = 2**5     # batch size
loss_fn = nn.L1Loss()       # Specify the loss function (nn.L2Loss for NMSE; nn.L1Loss for NMAE)
non_linear_transform = True
RS_inds = np.arange(6,12) - 1     # Column numbers of remote sensing data  in the textfile


# Prepare data for LSTM regression
main_dir = ''   # Specify the directory containing all the data
save_subdir = '' # Specify the subdir where all the trained models will be saved
fname = ''      # Specify the txt file containing raw data
fnameSplt = 'train_test_split.txt'
   
COMIDs, datenums_model, model_data = readRawData(main_dir, fname)

# Read train test split data
trn_tst_splt = TrainTestDates(main_dir, fnameSplt)

# Create trainign and testing data
train_data, test_data, COMID_sr, stdy_list = TrainingTestingData(COMIDs, model_data, trn_tst_splt, datenums_model, Lseq)
stdy_list = torch.from_numpy(stdy_list).float()

# Compute mean and standard deviation of each column to be used for standardization
mean_data = np.vstack(train_data)
meanx = np.nanmean(mean_data[:, 1:], axis=0)
stdx = np.nanstd(mean_data[:,1:], axis=0)
stdx[stdx==0] = 0.001
meanx = torch.from_numpy(meanx).float()
stdx = torch.from_numpy(stdx).float()

# Determine the input dimension
input_dim = len(meanx)
hidden_dim_mtan = input_dim

# LSTM formatting of training and testing data
train_final, test_final, train_inds, test_inds = Tesorformatting(COMID_sr, train_data, test_data, Lseq)

# Create data loaders
train_dataset = CustomData(train_final, meanx, stdx, stdy_list, train_inds, Lseq)
test_dataset = CustomData(test_final, meanx, stdx, stdy_list, test_inds, Lseq)
trn_sampler = SubsetRandomSampler(train_inds)
train_dataloader = DataLoader(train_dataset, batch_size = N, sampler=trn_sampler, collate_fn=CustomCollate_fn)

tst_sampler = SubsetSampler(test_inds)
test_dataloader = DataLoader(test_dataset, batch_size = N, sampler=tst_sampler, collate_fn=CustomCollate_fn)
nse_ts_list, ypred_list, yobs_list, test_COM_inds = [], [], [], []

# Model training
nse_ts_list, ypred_list, yobs_list, test_COM_inds = [], [], [], []
for seed in range(nseeds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # instantiate the model class
    lstm = mTANLSTMModel(input_dim, hidden_dim_lstm, n_lstm_layers, hidden_dim_mtan, embed_dim_mtan, num_heads, Lseq, output_dim, non_linear_transform, RS_inds)
    lstm.cuda()

    # define loss and optimizer
    lossfn_train = loss_fn
    #lossfn_train = CustomLoss(quant)
    optimizer = torch.optim.Adam(lstm.parameters(), lr = lrate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=optim_lr)

    # fix the number of epochs and start model training
    loss_tr_list = []
    loss_vl_list = []
    model_state = []
    nse_sv = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss_tr, state = train_mod(train_dataloader, lstm, lossfn_train, optimizer)
        model_state.append(copy.deepcopy(state))
        loss_tr_list.append(loss_tr)
        torch.cuda.empty_cache()
        print(loss_tr)
        scheduler.step()

        nse_ts, ypred, yobs, com_inds, stdy_out = test_mod(test_dataloader, lstm, lossfn_train)
        nse_sv.append(nse_ts)
    lstm = mTANLSTMModel(input_dim, hidden_dim_lstm, n_lstm_layers, hidden_dim_mtan, embed_dim_mtan, num_heads, Lseq, output_dim, non_linear_transform, RS_inds)
    lstm.cuda()
    lstm.load_state_dict(model_state[-1], strict = True)
    lossfn_test = loss_fn
    #lossfn_test = CustomLoss(quant)
    nse_ts, ypred, yobs, com_inds, stdy_out = test_mod(test_dataloader, lstm, lossfn_test)
    stdy_out = stdy_out.cpu().detach().numpy()
    ypred = [stdy_out[ind]*ypred[ind][0].cpu().detach().numpy() for ind in range(len(ypred))]
    yobs = [stdy_out[ind]*yobs[ind][0].cpu().detach().numpy() for ind in range(len(yobs))]
    com_inds = com_inds.cpu().detach().numpy().astype('int')
    
    ypred_list.append(ypred)
    yobs_list.append(yobs)
    test_COM_inds.append(com_inds)

    # save model
    sname = 'LSTM_state' + str(seed)
    filename = os.path.join(main_dir, save_subdir, sname)
    torch.save(lstm.state_dict(), filename)
    
    del lstm, optimizer, lossfn_train, lossfn_test
ypred = 0
for y in ypred_list: ypred += np.array(y)
ypred = ypred/nseeds
yobs = np.array(yobs)
com_inds = np.array(com_inds)
nse = computeNSE(yobs, ypred)

# write data to textfiles
NSE=[]
for com_ind in np.unique(com_inds):
    ind = np.nonzero(com_inds==com_ind)[0]
    yobs_tmp, ypred_tmp, comid = yobs[ind], ypred[ind], COMID_sr[com_ind]
    nse = computeNSE(yobs_tmp, ypred_tmp)
    NSE.append([comid, nse])

    
    sname = 'obs_pred_' + str(comid) + '.txt'
    filename = os.path.join(main_dir, save_subdir, sname)
    fid = open(filename, 'w')
    fid.write('NSE = ' + str(nse) + '\n')
    fid.write('Observed\tPredicted\n')
    for wind in range(len(yobs_tmp)):
        fid.write('%f\t%f\n'%(yobs_tmp[wind], ypred_tmp[wind]))
    fid.close()