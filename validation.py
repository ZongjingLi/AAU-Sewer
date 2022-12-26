import torch
import torch.nn as nnn

from model import *
from dataloader import *

import argparse

import numpy as np

parser = argparse.ArgumentParser()

from tqdm import tqdm

def validate_model(model,dataset):
    dataloader =  DataLoader(dataset,batch_size = 16)
    all_count = 0.0
    acc_count = 0.0
    for sample in tqdm(dataloader):
        data = sample[0];label = sample[1]
        data = torch.tensor(data).permute([0,2,1])
        logsmx,_,_ = model(data)   

        for i in range(logsmx.shape[0]):
            all_count += 1

            gt_label = label[i]
            predict_idx = np.argmax(logsmx[i].detach().numpy())
            #print(gt_label,predict_idx,gt_label==predict_idx)

            if (gt_label==predict_idx):acc_count += 1
    
    print("accuracy:{} {}/{}".format(acc_count/ all_count,acc_count,all_count))

if __name__ == "__main__":
    print("start the validation process")
    model = torch.load("checkpoints/point_net.ckpt",map_location = "cpu")
    #model = torch.load("point_net.ckpt",map_location = "cpu")
    aau_syn =  AAUSewer("train","synthetic")
    aau_real = AAUSewer("train","real")

    validate_model(model,aau_syn)