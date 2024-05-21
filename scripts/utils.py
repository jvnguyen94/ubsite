
#%%
# import csv
import glob
import numpy as np
#import tensorflow as tf
from sklearn.metrics import average_precision_score

from Bio import SeqIO


import pandas as pd
import argparse

import re
import os
import requests
from Bio.PDB.PDBParser import PDBParser
import urllib.request
from Bio.PDB import PDBList

from biotoolbox.structure_file_reader import build_structure_container_for_pdb
from biotoolbox.contact_map_builder import DistanceMapBuilder


def read_fasta(fn_fasta):
    aa = set(['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V',
             'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'])
    prot2seq = {}
    if fn_fasta.endswith('gz'):
        handle = gzip.open(fn_fasta, "rt")
    else:
       handle = open(fn_fasta, "rt")

    for record in SeqIO.parse(handle, "fasta"):
        seq = str(record.seq)
        prot = record.id
        pdb, chain = prot.split('_') if '_' in prot else prot.split('-')
        prot = pdb.upper() + '-' + chain
        #if len(seq) >= 60 and len(seq) <= 1000:
         #   if len((set(seq).difference(aa))) == 0:
        prot2seq[prot] = seq

    return prot2seq
  
def make_distance_maps(pdbfile, chain=None, sequence=None):
    """
    Generate (diagonalized) C_alpha and C_beta distance matrix from a pdbfile
    """
    pdb_handle = open(pdbfile, 'r')
    structure_container = build_structure_container_for_pdb(pdb_handle.read(), chain).with_seqres(sequence)
    # structure_container.chains = {chain: structure_container.chains[chain]}

    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)  # start with CA distances
    ca = mapper.generate_map_for_pdb(structure_container)
    cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)
    pdb_handle.close()

    return ca.chains, cb.chains


def retrieve_pdb(pdb, chain, chain_seqres, pdir):
    pdb_list = PDBList()
    pdb_list.retrieve_pdb_file(pdb, pdir=pdir)
    ca, cb = make_distance_maps(
        pdir + '/' + pdb + '.cif', chain=chain, sequence=chain_seqres)

    return ca[chain]['contact-map'], cb[chain]['contact-map']


def write_annot_npz(prot, prot2seq=None, out_dir=None):
    """
    Write to *.npz file format.
    """
    pdb, chain = prot.split('-')
    print('pdb=', pdb, 'chain=', chain)
    try:
        A_ca, A_cb = retrieve_pdb(pdb.lower(), chain, prot2seq[prot], pdir=os.path.join(
            out_dir, 'tmp_PDB_files_dir'))
        np.savez_compressed(os.path.join(out_dir, prot),
                            C_alpha=A_ca,
                            C_beta=A_cb,
                            seqres=prot2seq[prot],
                            )
    except Exception as e:
        print(e)
    

#%%
#parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('-seqres', type=str, default='./d/seq.fasta', help="PDB chain seqres fasta.")
    
prot2seq = read_fasta("./d/seq.fasta")
out_dir="./d/"
print ("### number of proteins with seqres sequences: %d" % (len(prot2seq)))
to_be_processed = set(prot2seq.keys())
for prot in to_be_processed:
    write_annot_npz(prot, prot2seq=prot2seq, out_dir=out_dir)
    
    #folder_path = "/Users/nguyjust/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/ub_site_pred/"
    #folder_path = "/Volumes/Justine-Mac/"


    ## Read in Ms fasta
    #ms100_fa = load_FASTA(folder_path+"data/ms_100.fasta")

    ## Separate MS fasta by ID and seq
    #ms100_id = ms100_fa[0]
    
    #ms100_seq = ms100_fa[1]
    #print("starting")
    
# %%
