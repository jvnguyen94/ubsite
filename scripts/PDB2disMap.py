#!/usr/bin/myenv python
import gzip
from biotoolbox.structure_file_reader import build_structure_container_for_pdb
from biotoolbox.contact_map_builder import DistanceMapBuilder
from Bio.PDB import PDBList
from Bio import SeqIO

from functools import partial
import numpy as np
import argparse
import csv
import os

#%%
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
        if len(seq) >= 60 and len(seq) <= 1000:
            if len((set(seq).difference(aa))) == 0:
                prot2seq[prot] = seq

    return prot2seq


def make_distance_maps(pdbfile, chain=None, sequence=None):
    """
    Generate (diagonalized) C_alpha and C_beta distance matrix from a pdbfile
    """
    pdb_handle = open(pdbfile, 'r')
    structure_container = build_structure_container_for_pdb(
        pdb_handle.read(), chain).with_seqres(sequence)
    # structure_container.chains = {chain: structure_container.chains[chain]}

    # start with CA distances
    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)
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


# %%
write_annot_npz(prot, prot2seq=prot2seq, out_dir=out_dir)
