# %%

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

print("Import Libraries & Set up directory")

folder_path = "/Users/nguyjust/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/ubsite/"


# %%
# list of 20 proteins
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Dictionary for getting Residue symbols
ressymbl = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

# %%
# def check_symmetric(a, rtol=1e-05, atol=1e-08):
#    print(np.allclose(a, a.T, rtol=rtol, atol=atol))


def file_names_funct(root_dir):

    file_name_path = [str(root_dir)+str(f)
                      for f in os.listdir(root_dir) if not f.startswith('.')]
    fn = [f for f in file_name_path if str(f).endswith(".pdb")]

    return fn


file_names = file_names_funct(folder_path+'alpha_pred/')

# file_names = [x for x in file_names if x[:-4] == '.pdb']
psp_df = pd.read_csv(folder_path + 'data/psp_info.tsv',
                     sep="\t", low_memory=False)

ub_list = psp_df[['acc_id', 'ub_mod_loc']].copy()
ub_list.rename(columns={'acc_id': 'protein',
          'ub_mod_loc': 'site'}, inplace=True)


# %%

def process(self):

        data_list = []
        count = 0

        for file in tqdm(file_names):
                #    if (pathlib.Path(file).suffix == ".pdb"):
                struct = get_structure(file)
                # print (struct)
                seq = self._get_sequence(struct)
                # print(seq)
                # print(self._get_SeqVecEmbedder(seq))
                # node features extracted
                node_feats = self._get_one_hot_symbftrs(seq)
                # print(node_feats)
                # edge-index extracted
                mat = self._get_adjacency(file)
                # print(mat)

                edge_index = self._get_edgeindex(file, mat)
                # print(f'Node features size :{torch.Tensor.size(node_feats)}')
                # print(f'Matrix size :{mat.shape}')
                # create data object
                # ADD Y = TARGET UB SITE AS binary fts (0./1)
                pos = ub_list.loc[ub_list['protein']
                    == self._get_id(file), 'site']
                # print(pos)
                ub_label = self._get_ubsite(seq, pos)
                # print(ub_label)
                data = Data(x=torch.tensor(node_feats, dtype=torch.float32),
                            y=torch.tensor(ub_label, dtype=torch.float32),
                            edge_index=torch.tensor(edge_index, dtype=torch.long))
                # print(data)
                count += 1
                print(count)
                # print("asdf")
                # data_list.append(data)
                # self.processed_file_names.append(
                #     folder_path + "processed/" + str(ub_list[protein])
                # torch.save(data, "./d/processed/" + f'data_{count}.pt')

            #return data_list

    # get structure from a pdb file
        # Uses biopython
    def _get_structure(self, file):
        parser = PDBParser()
        structure = parser.get_structure(self._get_id(file), file)
        return structure

        # Function to get sequence from pdb structure
        # Uses structure made using biopython
        # Those residues for which symbols are U / X are converted into A

        
    def _get_sequence(self, structure):
        sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in ressymbl.keys():
                        sequence = sequence + ressymbl[residue.get_resname()]
        return sequence
        # One hot encoding for symbols

    def _get_ubsite(self,sequence, position):
        n = len(sequence)
        ub_array = np.zeros((n,))
        ub_array[position] = 1
        
        return ub_array

    def _get_one_hot_symbftrs(self, sequence):
        one_hot_symb = np.zeros((len(sequence), len(pro_res_table)))
        row = 0
        for res in sequence:
            col = pro_res_table.index(res)
        one_hot_symb[row][col] = 1
        row += 1
        return torch.tensor(one_hot_symb, dtype=torch.float)
    
    def _get_adjacency(self, file):
        
        ## Check how I can look into side chains vs AlphaChains in terms of close proximity (is it avg distance or min distance between two residues?)
        edge_ind =[]
        molecule = bg.Pmolecule(file)
        network = molecule.network()
        mat = nx.adjacency_matrix(network)
        m = mat.todense()
        return m
    # get adjacency matrix in coo format to pass in GCNN model

    def _get_edgeindex(self, file, adjacency_mat):
        edge_ind = []
        m = self._get_adjacency(file)
        # check_symmetric(m, rtol=1e-05, atol=1e-08)

        a = np.nonzero(m > 0)[0]
        b = np.nonzero(m > 0)[1]
        edge_ind.append(a)
        edge_ind.append(b)
        return torch.tensor(np.array(edge_ind), dtype=torch.long)
    # Residue features calculated from pcp_dict

    def _get_res_ftrs(self, sequence):
        res_ftrs_out = []
        for res in sequence:
            res_ftrs_out.append(pcp_dict[res])
        res_ftrs_out = np.array(res_ftrs_out)
        # print(res_ftrs_out.shape)
        return torch.tensor(res_ftrs_out, dtype=torch.float)
    def _get_id(self, pdb_file):
        pdb_id = pdb_file.split(".")[-2].split('/')[-1]
        #print(pdb_id)
        return pdb_id

    # total features after concatenating one_hot_symbftrs and res_ftrs

    def _get_node_ftrs(self, sequence):
        one_hot_symb = one_hot_symbftrs(sequence)
        res_ftrs_out = res_ftrs(sequence)
        return torch.tensor(np.hstack((one_hot_symb, res_ftrs_out)), dtype=torch.float)

    def _get_id(self, pdb_file):
        pdb_id = pdb_file.split(".")[-2].split('/')[-1]
        #print(pdb_id)
        return pdb_id

    def _get_SeqVecEmbedder(self, seq):

        embedder = SeqVecEmbedder()
        embedding = embedder.embed(seq)
        protein_embd = torch.tensor(embedding).sum(dim=0)  # Vector with shape [L x 1024]
        np_arr = protein_embd.cpu().detach().numpy()
        
        return np_arr
    
    def _get_ProtBertEmbedder(self, seq):

        embedder = ProtTransBertBFDEmbedder()
        embedding = embedder.embed(seq)
        protein_embd = torch.tensor(embedding).sum(dim=0)  # Vector with shape [L x 1024]
        np_arr = protein_embd.cpu().detach().numpy()
        
        return np_arr

# %%
        """_summary_
        ## masking some of the nodes in the training sites

        """


train_files = ['./d/processed/data_1.pt', './d/processed/data_2.pt'] #file_names[0:2]
#test_files = #file_names[2:4]

train_dataset = ProteinDataset(ids=train_files)
# test_dataset = ProteinDataset(ids=test_files)

# train_dataset.process()
# test_dataset.process()
#data = data_class.process()



 #%%
import os
import numpy as np
from torch.utils.data import Dataset as Dataset_n
seed = 42
torch.manual_seed(seed)
# print(math.floor(0.8 * size))
# Make iterables using dataloader class
#trainset, validset = torch.utils.data.random_split(
#    data_class, [2, 2])
# print(trainset[0])
trainloader = DataLoader(dataset=train_dataset, batch_size = 2,  num_workers=0)
#testloader = DataLoader(dataset=test_dataset, batch_size = 2, num_workers=0)
# print("Length")
# print(len(trainloader))
# print(len(testloader))

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
#criterion = torch.nn.CrossEntropyLoss()
crit = torch.nn.BCELoss(weight=torch.tensor([0.01, 1])) # target is between 0,1, yhat is unnormalized logits

#%%
from sklearn.metrics import roc_auc_score

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


#def test():
#    model.eval()
#    out = model(trainloader.x, trainloader.edge_index)
#    pred = out.argmax(dim=1)
#    test_correct = pred[testloader.y]
#    #test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
#    return test_correct 



for epoch in range(1000):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


# %%
data = next(iter(trainloader))

yhat = model(data.x, data.edge_index)
yhat
# %%
plt.figure()
plt.hist(yhat.detach().cpu().numpy())
plt.show()
# %%

folder_path = "/Users/nguyjust/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/ubsite/data/"

print("Import Libraries & Set up directory")
ubi_data = pd.read_csv(folder_path + 'ub_site_complete.txt', 
                         sep="\t", low_memory=False)

## Select specific columns from mUbiSiDa
#data_mUbi_sel = data_mUbi[[0, 1, 9, 3, 6]]
#data_mUbi_sel.columns = ['acc_id','species','sequence', 'protein', 'ub_mod_loc']

#%%
import matplotlib.pyplot as plt


# %%




import pandas as pd

def generate_frequency_table(df, column_name):
    """
    This function takes a pandas DataFrame and a column name,
    and returns a frequency table of the values in that column.
    """
    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        frequency_table = df[column_name].value_counts().reset_index()
        frequency_table.columns = [column_name, 'Frequency']
        return frequency_table
    else:
        print(f"The column '{column_name}' does not exist in the DataFrame.")
        return None

# %%
ubi_site_per_prot =generate_frequency_table(ubi_data,"acc_id")
# %%

ubi_site_per_prot["Frequency"].hist()
plt.title(f'Histogram of Ub Sites per Protein (UniProt ID)')
plt.xlabel("# of Ub sites on Protein")
plt.ylabel('Frequency of Proteins')
plt.show()


# %%
freq_ub_site= ubi_site_per_prot["Frequency"].value_counts().reset_index()
freq_ub_site.columns = ["Num_Ub_Sites", 'Frequency']
plt.plot(freq_ub_site["Num_Ub_Sites"], freq_ub_site["Frequency"])
plt.xlabel("# of Unique Sites")
plt.ylabel('Frequency')
plt.show()
# %%
