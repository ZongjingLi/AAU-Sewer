import torch
import torch.nn as nn

from karanir.dklearn.nn.pnn import PointNetfeat

class UnivseralTransferModel(nn.Module):
    def __init__(self, config):
        super().__init__()