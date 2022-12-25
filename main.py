import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np

from model import *
from dataloader import *

import argparse 

from model    import *
from config   import *
from torch.utils.data import DataLoader

from train import *

opt_parser = argparse.ArgumentParser()
opt_parser.add_argument("--epoch",            default = 1000)
opt_parser.add_argument("--lr",               default = 2e-4)
opt_parser.add_argument("--batch_size",       default = 5)
opt_parser.add_argument("--source_dir",       default = "/content/MD_KITTI")
opt_parser.add_argument("--target_dir",       default = "/content/MD_KITTI")
opt_parser.add_argument("--update_steps",     default = 5)
opt_parser.add_argument("--transfer_batch",   default = 10)
opt_parser.add_argument("--transfer_samples", default = 100)
opt_parser.add_argument("--visualize_itrs",   default = 30)
opt_parser.add_argument("--tau",              default = 0.07)
opt_parser.add_argument("--omit_portion",     default = 0.3)
opt_parser.add_argument("--density_reduce",   default = 0.6)
opt = opt_parser.parse_args(args = [])


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    aau_syn = AAUSewer("train","synthetic")
    aau_real = AAUSewer("train","real")
    
    model = FeatureNet()
    
    train(model,aau_syn,opt)

    train_transfer(model,aau_syn,aau_real,opt)
    