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
        total_loss = 0
        itr = 0
        
        possible_index = list(range(len(dataset)))
        while len(possible_index) != 0:
            for sample in range(config.batch_size):
                sample_loc = np.random.choice(possible_index)
                possible_index.remove(sample_loc)

                itr += 1 # add one more iteration

        print("epoch: {} itr:{} total_loss:{}".format(epoch,itr,total_loss))

    return model

def train_transfer(model,source,target,config):
    print(config)