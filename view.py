from dataloader import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

aau_real = AAUSewer("train","real")
aau_syn = AAUSewer("train","synthetic")

r1 = aau_real[3]
s1 = aau_syn[3]

print(r1[0].shape,s1[0].shape)
print(r1[1],s1[1])

print(len(aau_real),len(aau_syn))

fig = plt.figure(figsize=(4,4))

r1 = s1