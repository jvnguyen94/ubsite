from os.path import isfile, join
import os.path as osp
from os import listdir
import math
import random
from torch_geometric.data import DataLoader
from bio_embeddings.embed import ProtTransBertBFDEmbedder, SeqVecEmbedder
from torch_geometric.data import Dataset, download_url, Data,  Batch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import pathlib
import biographs as bg
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
import os
import numpy as np
from torch.utils.data import Dataset as Dataset_n

print("Import Libraries & Set up directory")

folder_path = "/Users/nguyjust/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/ubsite/"


class ProteinDataset(Dataset):
    def __init__(self, ids):

        super().__init__()
        self.ids = ids

    def len(self):
        # Can iter through len and load through batches
        return len(self.ids)

    def get(self, idx):
        fname = self.ids[idx]
        return torch.load(fname)
        self.root = root
# %%
        """_summary_
        ## masking some of the nodes in the training sites

        """


train_files = [folder_path +'processed/onehot_O00461.pt',
                folder_path+'processed/onehot_O08997.pt'] 
                # file_names[0:2]
#test_files = #file_names[2:4]

train_dataset = ProteinDataset(ids=train_files)
# test_dataset = ProteinDataset(ids=test_files)

# train_dataset.process()
# test_dataset.process()
#data = data_class.process()




seed = 42
torch.manual_seed(seed)

trainloader = DataLoader(dataset=train_dataset, batch_size = 2,  num_workers=0)
#testloader = DataLoader(dataset=test_dataset, batch_size = 2, num_workers=0)


# %%

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


model = GCN(in_channels = 20, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
crit = torch.nn.BCELoss(weight=torch.tensor([0.01, 1])) # target is between 0,1, yhat is unnormalized logits


def train():
    
    model.train()
    losses = 0
    for batch_idx, data in enumerate(trainloader): 
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        pos_idx = (data.y == 1).nonzero(as_tuple=True)[0]
        neg_idx = (data.y == 0).nonzero(as_tuple=True)[0]
        neg_idx = neg_idx[torch.randint(0, len(neg_idx), size=(100,))]
        idx = torch.cat((pos_idx, neg_idx), dim=-1)
        loss = crit(out.squeeze()[idx], data.y[idx])
        loss.backward()
        optimizer.step()
        losses += loss.item() # .detach 

    return losses / len(trainloader)


for epoch in range(1000):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

data = next(iter(trainloader))

yhat = model(data.x, data.edge_index)
yhat

plt.figure()
plt.hist(yhat.detach().cpu().numpy())
plt.show()


# def test():
#    model.eval()
#    out = model(trainloader.x, trainloader.edge_index)
#    pred = out.argmax(dim=1)
#    test_correct = pred[testloader.y]
#    #test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
#    return test_correct

# %%
