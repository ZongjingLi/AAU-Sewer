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

print('input')
N = 2000
pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(N)
# fit to unit cube
pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
          center=pcd.get_center())
pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
o3d.visualization.draw_geometries([pcd])

print('voxelization')
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.05)
o3d.visualization.draw_geometries([voxel_grid])