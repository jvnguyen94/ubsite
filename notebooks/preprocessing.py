
#%%
######################################################################################
# Libraries
######################################################################################
#from os.path import isfile, join
from os import listdir
#import math
#import random
from bio_embeddings.embed import ProtTransBertBFDEmbedder, SeqVecEmbedder
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os
#import glob
from tqdm import tqdm
#import pathlib

import biographs as bg
#from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset, download_url, Data,  Batch

#%%
######################################################################################
# Functions
######################################################################################
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Dictionary for getting Residue symbols
ressymbl = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

# get structure from a pdb file


def get_structure(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure(get_id(pdb_file), pdb_file)
    return structure


def get_sequence(structure):
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in ressymbl.keys():
                    sequence = sequence + ressymbl[residue.get_resname()]
    return sequence


def get_ubsite(sequence, df):
    n = len(sequence)
    ub_array = np.zeros((n,))
    ub_sites = df[df['protein'] == get_id(ii)]['site'].tolist()
    ub_sites = [ii-1 for ii in ub_sites]
    ub_array[ub_sites] = 1

    return ub_array

# One hot encoding for symbols


def get_one_hot_symbftrs(sequence):
    one_hot_symb = np.zeros((len(sequence), len(pro_res_table)))
    row = 0
    for res in sequence:
        col = pro_res_table.index(res)
    one_hot_symb[row][col] = 1
    row += 1
    return torch.tensor(one_hot_symb, dtype=torch.float)


def get_adjacency(pdb_file):

    # Check how I can look into side chains vs AlphaChains in terms of close proximity (is it avg distance or min distance between two residues?)
    edge_ind = []
    molecule = bg.Pmolecule(pdb_file)
    network = molecule.network()
    mat = nx.adjacency_matrix(network)
    m = mat.todense()
    return m
# get adjacency matrix in coo format to pass in GCNN model


def get_edgeindex(pdb_file, adjacency_mat):
    edge_ind = []
    m = get_adjacency(pdb_file)
    # check_symmetric(m, rtol=1e-05, atol=1e-08)

    a = np.nonzero(m > 0)[0]
    b = np.nonzero(m > 0)[1]
    edge_ind.append(a)
    edge_ind.append(b)
    return torch.tensor(np.array(edge_ind), dtype=torch.long)


def get_id(pdb_file):

    return pdb_file.split("/")[-1].split('.')[-1]


def get_id(pdb_file):
    pdb_id = pdb_file.split(".")[-2].split('/')[-1]
    # print(pdb_id)
    return pdb_id


def get_SeqVecEmbedder(seq):

    embedder = SeqVecEmbedder()
    embedding = embedder.embed(seq)
    protein_embd = torch.tensor(embedding).sum(
        dim=0)  # Vector with shape [L x 1024]
    np_arr = protein_embd.cpu().detach().numpy()

    return np_arr


def get_ProtBertEmbedder(seq):

    embedder = ProtTransBertBFDEmbedder()
    embedding = embedder.embed(seq)
    protein_embd = torch.tensor(embedding).sum(
        dim=0)  # Vector with shape [L x 1024]
    np_arr = protein_embd.cpu().detach().numpy()

    return np_arr


def get_file_names(root_dir):

    file_name_path = [str(root_dir)+str(f)
                    for f in os.listdir(root_dir) if not f.startswith('.')]
    fn = [f for f in file_name_path if str(f).endswith(".pdb")]

    return fn


# %%
######################################################################################
# Main
######################################################################################
if __name__ == "__main__":

    # Main folder where repository lives
    #folder_path = "/Users/nguyjust/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/ubsite/"
    folder_path = "/Users/nguyjust/Documents/ubsite/"
    file_names = get_file_names('/Users/nguyjust/Documents/af_struc/')

    # Read in information for protein for site specific data
    psp_df = pd.read_csv(folder_path + 'data/psp_info.tsv',
                        sep="\t", low_memory=False)
    psp_df = pd.read_csv(folder_path + 'data/psp_info.tsv',
                        sep="\t", low_memory=False)
    ub_list = psp_df[['uniprot_id', 'ub_mod_loc']].copy()
    ub_list.rename(columns={'uniprot_id': 'protein',
                            'ub_mod_loc': 'site'}, inplace=True)

    ub_list.rename(columns={'uniprot_id': 'protein',
                'ub_mod_loc': 'site'}, inplace=True)
    count = 0

    for ii in tqdm(file_names):
        structure = get_structure(ii)
        seq = get_sequence(structure)

        ft_embed = "onehot"

        if ft_embed == "onehot":
            node_feats = get_one_hot_symbftrs(seq)
        elif ft_embed == "seqvec":
            node_feats = get_SeqVecEmbedder(seq)
        elif ft_embed == "protbert":
            node_feats = get_ProtBertEmbedder(seq)
        else:
            pass

            mat = get_adjacency(ii)
            # print(mat)

            ub_label = get_ubsite(seq, ub_list)
            edge_index = get_edgeindex(ii, mat)

            # print(ub_label)
            data = Data(x=torch.tensor(node_feats, dtype=torch.float32),
                        y=torch.tensor(ub_label, dtype=torch.float32),
                        edge_index=torch.tensor(edge_index, dtype=torch.long))

            torch.save(data, "../processed/" + ft_embed + f'_{get_id(ii)}.pt')
            # print(data)

            count += 1

# %%


# %%

