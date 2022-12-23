import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import argparse 
from opt import *

def train(model,dataset,config):

    # setup the optimizer and lr    
    optim = torch.optim.Adam(model.parameters(), lr = config.lr)

    for epoch in range(config.epoch):
        for sample in range(5):
            return 0

    return model

if __name__ == "__main__":
    train(0,0,config)