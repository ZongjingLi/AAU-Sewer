import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np
import argparse 

from model    import *
from config   import *
from dataloader import *
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset

from train import *
from utils import *

opt_parser = argparse.ArgumentParser()
opt_parser.add_argument("--epoch",            default = 1000)
opt_parser.add_argument("--lr",               default = 2e-4)
opt_parser.add_argument("--batch_size",       default = 32)
opt_parser.add_argument("--update_steps",     default = 5)
opt_parser.add_argument("--transfer_batch",   default = 10)
opt_parser.add_argument("--transfer_samples", default = 100)
opt_parser.add_argument("--visualize_itrs",   default = 30)
opt_parser.add_argument("--tau",              default = 0.07)
opt_parser.add_argument("--omit_portion",     default = 0.3)
opt_parser.add_argument("--density_reduce",   default = 0.6)
opt_parser.add_argument("--transfer_name",    default = "coral")
opt = opt_parser.parse_args(args = [])

remote = 1
device = "cuda:0" if torch.cuda.is_available() else "cpu"
TransferName = opt.transfer_name
root = "/content/gdrive/MyDrive/AAU/" if remote else "AAU"


# [Create Model]
model = PointNetCls(k = 4)
model = model.to(device)
# create dataset

aau_syn_train =  AAUSewer("train","synthetic")
model = train(model, aau_syn_train,opt)

aau_syn_test =  AAUSewer("test","synthetic")
aau_syn_test.train_data = torch.tensor(np.load(root+"{}_syn_test.npy".format(TransferName))).float()

aau_real_test = AAUSewer("test","real")
aau_real_test.train_data = torch.tensor(np.load(root+"{}_real_test.npy".format(TransferName))).float()

aau_syn_train =  AAUSewer("train","synthetic")
aau_syn_train.train_data = torch.tensor(np.load(root+"{}_syn_train.npy".format(TransferName))).float()

aau_real_train = AAUSewer("train","real")
aau_real_train.train_data = torch.tensor(np.load(root+"{}_real_train.npy".format(TransferName))).float()

"""
[Setup]
"""
torch.save(model,"{}_eval_model.ckpt".format(opt.transfer_name))

source_dataset = torch.utils.data.ConcatDataset([aau_syn_train,aau_syn_test])
target_dataset = torch.utils.data.ConcatDataset([aau_real_train,aau_real_test])

k = 10
splits=KFold(n_splits=k,shuffle=True,random_state=42)

history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
eval(model,dataset = target_dataset)

overall = {"accuracy":[],"precision":[],"recall":[],"F1":[]}
for source_fold, (source_train_idx,source_val_idx) in enumerate(splits.split(np.arange(len(source_dataset)))):
    target_fold, (target_train_idx,target_val_idx) = next(enumerate(splits.split(np.arange(len(target_dataset)))))
    print('Fold {}'.format(source_fold + 1))

    source_train_sampler = SubsetRandomSampler(source_train_idx)
    source_test_sampler = SubsetRandomSampler(source_val_idx)
    source_train_loader = DataLoader(source_dataset, batch_size=opt.batch_size, sampler=source_train_sampler)
    source_test_loader = DataLoader(source_dataset, batch_size=opt.batch_size, sampler=source_test_sampler)


    target_train_sampler = SubsetRandomSampler(target_train_idx)
    target_test_sampler = SubsetRandomSampler(target_val_idx)
    target_train_loader = DataLoader(target_dataset, batch_size=opt.batch_size, sampler=target_train_sampler)
    target_test_loader = DataLoader(target_dataset, batch_size=opt.batch_size, sampler=target_test_sampler)

    acc, prec, recall, f1 = eval(model,target_test_loader)
    overall["accuracy"].append(acc)
    overall["precision"].append(prec)
    overall["recall"].append(recall)
    overall["F1"].append(f1)


print("\nAverage Metrics:")
for key in overall:
    print("{}:{}".format(key, np.array(overall[key]).mean()))