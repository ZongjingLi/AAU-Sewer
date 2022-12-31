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
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cuda:0"
opt_parser = argparse.ArgumentParser()
opt_parser.add_argument("--device",           default = device)
opt_parser.add_argument("--epoch",            default = 400)
opt_parser.add_argument("--lr",               default = 2e-4)
opt_parser.add_argument("--batch_size",       default = 16)
opt_parser.add_argument("--update_steps",     default = 250)
opt_parser.add_argument("--transfer_batch",   default = 10)
opt_parser.add_argument("--transfer_samples", default = 100)
opt_parser.add_argument("--visualize_itrs",   default = 30)
opt_parser.add_argument("--tau",              default = 0.07)
opt_parser.add_argument("--omit_portion",     default = 0.3)
opt_parser.add_argument("--density_reduce",   default = 0.6)
opt = opt_parser.parse_args(args = [])


class AAUSewer(Dataset):
    def __init__(self,split = "train",type = "real",l=None):
        super().__init__()
        self.split = split
        self.length = l
        self.train_data = []
        self.labels = []
        dataDir = "/content/gdrive/MyDrive/AAU"
        path = os.path.join(dataDir, "{}_{}.h5".format("{}ing_pointcloud_hdf5".format(split), type))
        #path = "/content/training_pointcloud_hdf5_synthetic.h5"
        #path = "/content/testing_pointcloud_hdf5_synthetic.h5"
        print(path)
        with h5py.File(path, 'r') as hdf:          
            if split == "train":
                partitions = ["Training"]
            else:
                partitions = ["Testing"]

            for partition in partitions:
                self.train_data = np.asarray(hdf[f'{partition}/PointClouds'][:])
                self.labels = np.asarray(hdf[f'{partition}/Labels'][:])
      

    def __len__(self):
      if self.length is not None:return self.length
      return self.labels.shape[0]

    def __getitem__(self,index):
        return torch.tensor(self.train_data[index]).cuda(),self.labels[index]

def train(model,dataset,config):

    # setup the optimizer and lr    
    optim = torch.optim.Adam(model.parameters(), lr = config.lr)
    
    dataloader = DataLoader(dataset,batch_size = config.batch_size)
    history = []
    for epoch in range(config.epoch):
        total_loss = 0
        for sample in tqdm(dataloader):
            data = sample[0].to(config.device);label = sample[1]
            data = data.permute([0,2,1])

            logsmx,_,_ = model(data)   

            loss = 0
            for i in range(label.shape[0]):
                loss -= logsmx[i][label[i]]
            total_loss += loss.cpu().detach().numpy()

            
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # augument the data to get the result
            data = sample[0].to(config.device);label = sample[1]
            data = data.permute([0,2,1])

            reduce_number = 300
            avail= list(range(1024))

            n = np.random.randint(800,1000)

            data = data[:,:n]

            logsmx,_,_ = model(data)   

            loss = 0
            for i in range(label.shape[0]):
                loss -= logsmx[i][label[i]]
            total_loss += loss.cpu().detach().numpy()

            
        torch.save(model,"point_net_ckpt.ckpt")
        history.append(total_loss)
        print("epoch: {} total_loss:{}".format(epoch,total_loss))

        validate_model(model,aau_syn_test)
        validate_model(model,aau_syn_train)
        validate_model(model,aau_real_test)
        validate_model(model,aau_real_train) 

        plt.plot(history)
        plt.pause(0.001)
        plt.cla()

    return model

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat 

    def pred(self,x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        feature = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(feature)
        return F.log_softmax(x, dim=1), feature
    
def calculate_features(model,x):
    x, trans, trans_feat = model.feat(x)
    x = F.relu(model.bn1(model.fc1(x)))
    features = F.relu(model.bn2(model.dropout(model.fc2(x))))
    labels = model.fc3(features)
    labels = F.log_softmax(labels, dim=1)
    return labels,features


from tqdm import tqdm

def train_transfer(model,source_data,target_data,update_step = opt.update_steps):
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4)

    batch_size = 32

    # generate pesudo labels on the target data set
    source_loader = DataLoader(source_data,batch_size = batch_size,shuffle=True)
    target_loader = DataLoader(target_data,batch_size = batch_size,shuffle=True)
    
    history = []
    for k in range(opt.update_steps):
        # single time update
        total_loss = 0
        device = "cuda:0"
        
        for target_batch in target_loader:
            for source_batch in tqdm(source_loader):
                working_loss = 0
                # calculate the transfer batch
                _,rt_labels = target_batch
                raw_labels,features = calculate_features(model,target_batch[0].permute([0,2,1]).to(device))

                _,gt_labels = source_batch
                data = source_batch[0].permute([0,2,1]).to(device)
                source_labels,source_features = calculate_features(model,data)

                loss = 0
                for i in range(source_labels.shape[0]):
                    loss -= 1 * source_labels[i][gt_labels[i]]
                
                n = np.random.randint(800,1000)

                data = data[:,:n]

                logsmx,_,_ = model(data)   
                for i in range(source_labels.shape[0]):
                    loss -= 1 * logsmx[i][gt_labels[i]]

                for i in range(raw_labels.shape[0]):
                    loss -= raw_labels[i][rt_labels[i]]
                working_loss += loss

                optimizer.zero_grad()
                working_loss.backward()
                optimizer.step()

                #working_loss -= raw_labels[i][label_i]- raw_labels[j][label_j]
                # calculate contrastive loss for each label:
                for _ in range(4):
                    # inter domain loss (source)
                    inter_source_loss = 0
                    logp_source = 0
                    for i in range(source_labels.shape[0]):
                        for j in range(source_labels.shape[0]):
                            label_i = np.argmax(source_labels[i].cpu().detach().numpy())
                            label_j = np.argmax(source_labels[j].cpu().detach().numpy())
                            flag = label_i == label_j

                            if gt_labels[i] == label_i and gt_labels[j] == label_j:
      
                              if not flag and i!=j:   
                                  logp_source -= torch.cosine_similarity(source_features[i] , source_features[j],0).exp()
                              if i!=j and flag:
                                  logp_source += torch.cosine_similarity(source_features[i] , source_features[j],0).exp()
                    inter_source_loss = 0 - logp_source

                    # inter domain loss (target)
                    inter_target_loss = 0
                    logp = 0
                    for i in range(rt_labels.shape[0]):
                        for j in range(rt_labels.shape[0]):
                            label_i = np.argmax(raw_labels[i].cpu().detach().numpy())
                            label_j = np.argmax(raw_labels[j].cpu().detach().numpy())
                            flag = label_i == label_j
                      
                            if rt_labels[i] == label_i and rt_labels[j] == label_j:
  
                              if not flag and i!=j:   
                                logp -= torch.cosine_similarity(features[i] , features[j],0).exp()
                              if i!=j and flag:
                                logp += torch.cosine_similarity(features[i] , features[j],0).exp()
                            #working_loss -= raw_labels[i][label_i]- raw_labels[j][label_j]
                    inter_target_loss = 0 - logp
                    

                    # intra domain loss (source and target)
                    inter_domain_loss = 0
                    logp_transfer = 0
                    for i in range(raw_labels.shape[0]):
                        for j in range(source_labels.shape[0]):
                            label_i = np.argmax(raw_labels[i].cpu().detach().numpy())
                            label_j = np.argmax(source_labels[j].cpu().detach().numpy())
                            flag = label_i == label_j
                            #print(torch.linalg.norm(features[i] - features[j]))
                            #print(gt_labels.shape,rt_labels.shape)
                            try:
                                if rt_labels[i] == label_i and gt_labels[j] == label_j:
                                    if not flag and i!=j:   
                                      logp_transfer -= torch.cosine_similarity(features[i] , source_features[j],0).exp()
                                    if i!=j and flag:
                                      logp_transfer += torch.cosine_similarity(features[i] , source_features[j],0).exp()
                            except:print("namo")
                    inter_domain_loss = 0 - logp_transfer

                    working_loss += (inter_target_loss + inter_source_loss + inter_domain_loss) * 0.01


 
                total_loss += working_loss.cpu().detach().numpy()

        history.append(total_loss)
      
        plt.plot(history)
        plt.pause(0.001);plt.cla()
        print("transfer_step:{} loss:{}".format(k,total_loss))
        
        validate_model(model,aau_syn_test)
        validate_model(model,aau_real_test)
        validate_model(model,aau_real_train)
        torch.save(model,"transfer_model.ckpt")


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
    model = torch.load("checkpoints/point_net3.ckpt",map_location = "cpu")
    #model = torch.load("point_net.ckpt",map_location = "cpu")
    aau_syn =  AAUSewer("train","synthetic")
    aau_real = AAUSewer("train","real")

    validate_model(model,aau_syn)