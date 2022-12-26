from dataloader import *
from model    import *
from config   import *
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(model,dataset,config):

    # setup the optimizer and lr    
    optim = torch.optim.Adam(model.parameters(), lr = config.lr)
    dataloader = DataLoader(dataset,batch_size = config.batch_size)

    history = []
    for epoch in range(config.epoch):
        total_loss = 0
        for sample in tqdm(dataloader):
            data = sample[0];label = sample[1]
            data = torch.tensor(data).permute([0,2,1])
            logsmx,_,_ = model(data)   

            loss = 0
            for i in range(label.shape[0]):
                loss -= logsmx[i][label[i]]
            total_loss += loss.detach().numpy()

            
            loss.backward()
            optim.step()
            optim.zero_grad()
            history.append(loss.detach().numpy())

            torch.save(model,"point_net.ckpt")
        print("epoch: {} total_loss:{}".format(epoch,total_loss))

    return model

def train_transfer(model,source,target,config):
    optim = torch.optim.Adam(model.parameters(), config.lr) 
    dataloader = DataLoader(source,batch_size = config.batch_size)

    for epoch in range(config.epoch):
        total_loss = 0
        for sample in tqdm(dataloader):
            data = sample[0];label = sample[1]
            data = torch.tensor(data).permute([0,2,1])
            logsmx,_,_ = model(data)   

            loss = 0
            for i in range(label.shape[0]):
                loss -= logsmx[i][label[i]]
            total_loss += loss.detach().numpy()

            
            loss.backward()
            optim.step()
            optim.zero_grad()

            torch.save(model,"point_net.ckpt")
        print("epoch: {} total_loss:{}".format(epoch,total_loss))



