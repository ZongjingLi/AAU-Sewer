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
        
            #plt.plot(history)
            #plt.pause(0.001)
            #plt.cla()
            
            torch.save(model,"point_net.ckpt")
        print("epoch: {} total_loss:{}".format(epoch,total_loss))

    return model

def train_transfer(model,source,target,config):
    optim = torch.optim.Adam(model.parameters(), config.lr)
    
    for epoch in range(config.epoch):
        total_loss = 0
        itr = 0

        possible_index = list(range(len(source)))
        while len(possible_index) != 0:
            for sample in range(config.batch_size):
                sample_loc = np.random.choice(possible_index)
                possible_index.remove(sample_loc)

                itr += 1 # add one more iteration

        print("epoch: {} itr:{} total_loss:{}".format(epoch,itr,total_loss))


