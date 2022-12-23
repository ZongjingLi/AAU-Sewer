import os
from re import L
import h5py
import numpy as np

from torch.utils.data import DataLoader,Dataset

from config import *

dataDir = config.data_dir
hdf5Files = ["training_pointcloud_hdf5", "testing_pointcloud_hdf5"]
dataTypes = ["synthetic", "real"]
partitions = ["Training", "Validation"]

classLabels = {0:"Normal", 1:"Displacement", 2:"Brick", 3:"Rubber Ring"}

for h5 in hdf5Files:
    for dt in dataTypes:
        path = os.path.join(dataDir, "{}_{}.h5".format(h5, dt))

        with h5py.File(path, 'r') as hdf:          
            if h5 == "training_pointcloud_hdf5":
                partitions = ["Training", "Validation"]
            else:
                partitions = ["Testing"]

            for partition in partitions:
                data = np.asarray(hdf[f'{partition}/PointClouds'][:])
                labels = np.asarray(hdf[f'{partition}/Labels'][:])

                uniqueLabels, uniqueCounts = np.unique(labels, return_counts = True)
                print(f'\nFilepath: {path}')
                print(f'[{partition} Data]')
                print(f'Data Shape: {data.shape} | Type: {data[0].dtype}')
                print(f'Label Shape: {labels.shape} | Type: {labels[0].dtype}')
                print(f'Labels: {uniqueLabels} | Counts: {uniqueCounts}\n')

class AAUSewer(Dataset):
    def __init___(self,split = "train"):
        super().__init__()

    def __len__(self):return 0

    def __getitem__(self,index):
        return index