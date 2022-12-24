import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np

from model import *
from dataloader import *

import argparse 
from opt import *

from dataloader import *
from model    import *
from config   import *
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

def train(model,dataset,config):

    # setup the optimizer and lr    
    optim = torch.optim.Adam(model.parameters(), lr = config.lr)

    for epoch in range(config.epoch):
        for sample in range(config.batch_size):
            pass

    return model

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = RPN3D("Car",1,1).to(device)

    train(model,0,opt)