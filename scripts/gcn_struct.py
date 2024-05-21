#%%
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from Bio.PDB.PDBParser import PDBParser
import urllib.request
from Bio import SeqIO
import pandas as pd
## Read in contact map
# %%
def split_train_test(fasta_file):
    return None

# %%
def load_predicted_PDB(uniprot, fp):
    """
    .
    
    :param uniprot: 
    :param fp: 
    :return: 
    
    **from:
    """
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    #print("file path:" + fp+"alpha_pred/"+uniprot+".pdb")
    structure = parser.get_structure(uniprot, fp+uniprot+".pdb")
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(fp+uniprot+".pdb", 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one-two)

    return uniprot,distances, seqs[0]

# %%
import os

prot_dir = os.listdir("./d/")

prot_list = [prot_file.split(".")[0] for prot_file in prot_dir if prot_file.split(".")[1]=="pdb"]


[load_predicted_PDB(ii, "./d/") for ii in prot_list]

# %%
data = [['A0A5E8G9T8', 12], ['A0A5E8G9H8', 24]]
df = pd.DataFrame(data, columns=['prot', 'loc'])


# %%
# Install bio_embeddings using the command: pip install bio-embeddings[all]

from bio_embeddings.embed import SeqVecEmbedder
import numpy as np
import torch 

seq = 'MVTYDFGSDEMHD' # A protein sequence of length L

embedder = SeqVecEmbedder()
embedding = embedder.embed(seq)
protein_embd = torch.tensor(embedding).sum(dim=0) # Vector with shape [L x 1024]
np_arr = protein_embd.cpu().detach().numpy()


# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import pathlib
import math
import sklearn
import torch_optimizer as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from metrics import *

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.cuda("cpu")

import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
#from data_prepare import dataset, trainloader, testloader
from models import GCNN, AttGNN
from torch_geometric.data import DataLoader as DataLoader_n


def load_predicted_PDB(uniprot, fp):
    """
    .
    
    :param uniprot: 
    :param fp: 
    :return: 
    
    **from:
    """
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    #print("file path:" + fp+"alpha_pred/"+uniprot+".pdb")
    structure = parser.get_structure(uniprot, fp+uniprot+".pdb")
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(fp+uniprot+".pdb", 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one-two)

    return uniprot,distances, seqs[0]


# %%
import os

prot_dir = os.listdir("./d/")

prot_list = [prot_file.split(".")[0] for prot_file in prot_dir if prot_file.split(".")[1]=="pdb"]


[load_predicted_PDB(ii, "./d/") for ii in prot_list]

# %%
data = [['A0A5E8G9T8', 12], ['A0A5E8G9H8', 24]]
df = pd.DataFrame(data, columns=['prot', 'loc'])
# %%
