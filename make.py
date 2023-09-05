# Make Transfer Dataset (If it is not Available)
from transfertools.models import LocIT
from transfertools.models import TCA, CORAL
import torch
import torch.nn as nn
import numpy as np
from dataloader import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--transfer_name",  default = "tca")
opt = parser.parse_args()

TransferName = opt.transfer_name

if __name__ == "__main__":

    # [Create Aligned Dataset] (Test)
    aau_syn_test =  AAUSewer("test","synthetic")
    aau_real_test = AAUSewer("test","real")
    N = aau_syn_test.train_data.shape[0]
    M = aau_real_test.train_data.shape[0]
    print("Transfer from {} to {}".format(N,M))
    npt = 1024
    if TransferName == "coral":
      transfor = CORAL(scaling='standard') # [Transfer Model]
    if TransferName == "locit":
      transfor = LocIT()# [Transfer Model]
    if TransferName == "tca":
      transfor = TCA(n_components = 100 * 3 )# [Transfer Model]
    outputs = transfor.fit_transfer(aau_syn_test.train_data.reshape(N,npt*3), aau_real_test.train_data.reshape(M,npt*3))
    Xs_trans, Xt_trans = outputs
    aau_syn_test.train_data = torch.tensor(Xs_trans.reshape(N,100,3)).float()
    aau_real_test.train_data = torch.tensor(Xt_trans.reshape(M,100,3)).float()
    print(aau_syn_test.train_data.shape, aau_real_test.train_data.shape)

    np.save("{}_syn_test.npy".format(TransferName),aau_syn_test.train_data)
    np.save("{}_real_test.npy".format(TransferName),aau_real_test.train_data)

    # [Create Aligned Dataset] (Train)
    aau_syn_train =  AAUSewer("train","synthetic")
    aau_real_train = AAUSewer("train","real")
    N = aau_syn_train.train_data.shape[0]
    M = aau_real_train.train_data.shape[0]
    print("Transfer from {} to {}".format(N,M))

    #transfor = CORAL(scaling='standard') # [Transfer Model]
    outputs = transfor.fit_transfer(aau_syn_train.train_data.reshape(N,npt*3), aau_real_train.train_data.reshape(M,npt*3))
    Xs_trans, Xt_trans = outputs
    aau_syn_train.train_data = torch.tensor(Xs_trans.reshape(N,100,3)).float()
    aau_real_train.train_data = torch.tensor(Xt_trans.reshape(M,100,3)).float()
    print(aau_syn_train.train_data.shape, aau_real_train.train_data.shape)

    np.save("{}_syn_train.npy".format(TransferName),aau_real_test.train_data)
    np.save("{}_real_train.npy".format(TransferName),aau_real_train.train_data)