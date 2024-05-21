# %%
######################################################################################
# Libraries
######################################################################################
import urllib.request
import numpy as np
import pandas as pd
from Bio import SeqIO
import re
import os
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt


print("Import Libraries & Set up directory")

folder_path = "/Users/nguyjust/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/ubsite/"

#%%

# Data read in
fasta_seqs = list(SeqIO.parse(open(folder_path + 'raw_data/idmapping_2023_08_29.fasta'), 'fasta'))

print("Read in fasta file")

## Clean fasta ID names for quicker query
for ii in range(len(fasta_seqs)):
    fasta_seqs[ii].id = fasta_seqs[ii].id.split("|")[1] 
    
fasta_seq_df = pd.DataFrame(columns = ['id', 'seq'])

for ii in range(len(fasta_seqs)):
    fasta_seq_df.loc[len(fasta_seq_df.index)] = [fasta_seqs[ii].id, fasta_seqs[ii].seq]
    #fasta_seq_df.append({'id': fasta_seqs[0].id, 'seq': fasta_seqs[0].seq}, ignore_index=True)
#%%

fasta_seq_df['seq'] = fasta_seq_df['seq'].astype(str)
# %%
######################################################################################
# dbPTM
######################################################################################
print('dbPTM cleaning')

# Read in the dbPTM Dataset
data_dbptm = pd.read_csv(folder_path + 'raw_data/dbPTM/Ubiquitination',
                         sep="\t", low_memory=False, header=None, skiprows=[0, 1])

# Select specific columns from dbPTM
data_dbptm_sel = data_dbptm[[0, 1, 2, 5]]
data_dbptm_sel.columns = ['protein', 'acc_id', 'ub_mod_loc', 'seq']
data_dbptm_sel = data_dbptm_sel.dropna()


## Double check that all the sequences given have a lysine in the center of the 21-amino acid length
## RESULT: All sequences have a lysine so this check is passed

# data_dbptm_sel_k = []
# for ii in range(len(data_dbptm_sel)):
#     if data_dbptm_sel['seq'][ii][10] == "K":
#        data_dbptm_sel_k.append(data_dbptm_sel["acc_id"])
#%%
data_dbptm_valid_entry = data_dbptm_sel

for ii in range(len(data_dbptm_sel)):
    try:
        data = fasta_seq_df.loc[fasta_seq_df['id'] == data_dbptm_sel['acc_id'][ii]]
    #if str(data_dbptm_sel['seq'][ii]) in str(data.iloc[0]['seq']):
        s = re.sub('[^a-zA-Z]+', '', str(data_dbptm_sel['seq'][ii]))
        if s in str(data.iloc[0]['seq']):
     
        #data_dbptm_valid_entry = data_dbptm_valid_entry.append(
        #    data_dbptm_sel.iloc[ii])
        #print("yes {}".format(ii))
        #try:
        #    data_dbptm_valid_entry.loc[ii] = data_dbptm_sel.iloc[ii]
        #except:
            pass
        else:
        #print("Pass on {}".format(data_dbptm_sel['acc_id'][ii]))
            data_dbptm_valid_entry = data_dbptm_valid_entry.drop(index= [ii])
            print(ii)
    except:
        pass
  #%%
# Write out master info
#data_dbptm_valid_entry.to_csv(
#    folder_path + '/data/dbptm_info.tsv', sep='\t', index=False)

# Write out IDs to search
# data_dbptm_sel['acc_id'].to_csv(folder_path + '/data/dbptm_ids.txt', sep = '\t', index=False)

# Quick EDA
# Number of sites
len(data_dbptm_valid_entry.index)

# Number unique proteins
data_dbptm_valid_entry.groupby(['acc_id']).ngroups
# %%
# %%
######################################################################################
# PhosphoSitePlus
######################################################################################

# Read in the PhosphoSite Dataset
data_psp = pd.read_csv(folder_path + '/raw_data/PSP/Ubiquitination_site_dataset',
                       sep="\t", low_memory=False, skiprows=[0, 1, 2])

# Look at the header to check file integrity
# list(data_psp)
#%%
# Data with selected data columns
# PROTEIN, ACC_ID, ORGANISM, MOD_RSD
data_sel_psp = data_psp[['PROTEIN', 'ACC_ID',
                         'ORGANISM', 'MOD_RSD', 'SITE_+/-7_AA']].copy()

# Check that correct info is copied
# data_sel_psp.head()

# Clean the Ub location column
# ie: originally K##-ub, but just need to capture the location
data_sel_psp['ub_mod_loc'] = data_sel_psp.MOD_RSD.str.extract('(\d+)')
data_sel_psp = data_sel_psp.drop(['MOD_RSD'], axis=1)

# Check that only numbers were captured in the location
# data_sel_psp.head(100)
data_sel_psp_k = data_sel_psp

for ii in range(len(data_sel_psp)):
    if data_sel_psp['SITE_+/-7_AA'][ii][7] == "k":
        #data_sel_psp_k = pd.concat([data_sel_psp_k, data_sel_psp.iloc[ii]])
        pass
    else:
        print(ii)
        data_sel_psp_k = data_sel_psp_k.drop(index=[ii])
#%%
data_sel_psp_k['SITE_+/-7_AA'] = data_sel_psp_k['SITE_+/-7_AA'].str.upper()

data_psp_valid_entry = data_sel_psp_k

for ii in range(len(data_sel_psp_k)):
    try:
        data = fasta_seq_df.loc[fasta_seq_df['id']
                                == data_sel_psp_k['ACC_ID'][ii]]
    # if str(data_dbptm_sel['seq'][ii]) in str(data.iloc[0]['seq']):
        s = re.sub('[^a-zA-Z]+', '', str(data_sel_psp_k['SITE_+/-7_AA'][ii]))
        if s in str(data.iloc[0]['seq']):
            pass
        else:
            data_psp_valid_entry = data_psp_valid_entry.drop(index=[ii])
            print(ii)
    except:
        pass
#%%

# Clean up headers and write out
data_psp_valid_entry.columns = [x.lower()
                                for x in data_psp_valid_entry.columns]
data_psp_valid_entry = data_psp_valid_entry.rename(
    columns={"site_+/-7_aa": "seq"})
#%%
# Write out master info
#data_psp_valid_entry.to_csv(
#    folder_path + '/data/psp_info.tsv', sep='\t', index=False)

# Master information
# data_sel_psp.to_csv(folder_path + '/data/psp_info.txt', sep = '\t', index=False)

# UniProt IDs to search for full length sequences
# data_sel_psp['acc_id'].to_csv(folder_path + '/data/psp_ids.txt', sep = '\t', index=False)

# Quick EDA
# Number of sites
len(data_psp_valid_entry.index)
data_psp_valid_entry.groupby(['acc_id']).ngroups

psp_id = pd.DataFrame(list(set(data_psp_valid_entry['acc_id'])))
psp_id.to_csv(folder_path + '/data/psp_id.txt', sep='\t', index=False)

# %%
######################################################################################
# PLMD
######################################################################################
print('PLMD db cleaning')

# Read in the PLMD Dataset
data_PLMD = pd.read_csv(folder_path + '/raw_data/PLMD/Ubiquitination.txt',
                        sep="\t", low_memory=False, header=None)

# Select specific columns from PLMD
data_PLMD_sel = data_PLMD[[4, 1, 5, 2, 6]]
data_PLMD_sel.columns = ['protein',	'acc_id', 'organism', 'ub_mod_loc', "seq"]

# Check that rearranging data was successful
data_PLMD_sel.head()

# Master information
# data_PLMD_sel.to_csv(folder_path + '/data/plmd_info.txt', sep = '\t', index=False)

# Quick EDA
# Number of sites
len(data_PLMD_sel.index)
# Number of organisms
len(pd.unique(data_PLMD_sel['organism']))
# Number unique proteins
data_PLMD_sel.groupby(['protein', 'organism']).ngroups

# Create df with protein acc_id and seq for creating a fasta file!
#data_plmd_fasta = data_PLMD[[1, 6]]
#data_plmd_fasta.columns = ['acc_id', 'seq']

# Write fasta file
# with open(folder_path + '/data/plmd.fasta', 'w') as fh:
#    for i in range(data_plmd_fasta.shape[0]):
#        fh.write('>'+ str(data_plmd_fasta['acc_id'][i]) + '\n' +  str(data_plmd_fasta['seq'][i]) + '\n')
#%%
#data_PLMD_sel.to_csv(
#    folder_path + '/data/plmd_info.tsv', sep='\t', index=False)
# %%
plmd_seq = list(data_PLMD_sel['seq'])
plmd_pos = list(data_PLMD_sel["ub_mod_loc"])


for ii in range(len(plmd_pos)):
    pos = int(plmd_pos[ii]) -1
    if str(plmd_seq[ii])[pos] == "K":
        pass
    else:
        print(ii)
# %%


data = fasta_seq_df.loc[fasta_seq_df['id']
                        == "P31946"]
## site = 5, 11

# %%
######################################################################################
# mUbiSiDa
######################################################################################
print('mUbiSiDa cleaning')

# Read in the mUbiSiDa Dataset
data_mUbi = pd.read_csv(folder_path + '/raw_data/mUbiSiDa/data_2013_10_22.csv',
                        sep=",", low_memory=False, header=None, skiprows=[0, 1])

# Select specific columns from mUbiSiDa
data_mUbi_sel = data_mUbi[[0, 1, 9, 3, 6]]
data_mUbi_sel.columns = ['acc_id', 'species',
                         'sequence', 'protein', 'ub_mod_loc']
new_data_mUbi_sel = pd.DataFrame(data_mUbi_sel.ub_mod_loc.str.split(';').tolist(),
                                 index=[data_mUbi_sel.acc_id, data_mUbi_sel.protein]).stack()
new_data_mUbi_sel = new_data_mUbi_sel.reset_index([0, 'acc_id', 'protein'])
new_data_mUbi_sel.columns = ['acc_id', 'protein', 'ub_mod_loc']
# %%
data_mUbi_complete = new_data_mUbi_sel.merge(data_mUbi_sel, how="outer", on=['acc_id'])
data_mUbi_complete = data_mUbi_complete[data_mUbi_complete.columns[0:5]]
# %%
mUbi_seq = list(data_mUbi_complete['sequence'])
mUbi_pos = list(data_mUbi_complete["ub_mod_loc_x"])

fail=[]

for ii in range(len(mUbi_pos)):
    
    
    try:
        pos = int(mUbi_pos[ii]) - 1
        if str(mUbi_seq[ii])[pos] == "K":
            pass
        else:
            print(ii)
            fail.append(ii)
    except:
        pass
# %%

## Read in files

dbptm_df = pd.read_csv(folder_path+"data/dbptm_info.tsv",  sep="\t")
plmd_df = pd.read_csv(folder_path+"data/plmd_info.tsv",  sep="\t")
psp_df = pd.read_csv(folder_path+"data/psp_info.tsv",  sep="\t")
dbptm_df_2 = dbptm_df[['protein','acc_id','ub_mod_loc', 'seq']]
dbptm_df_2['organism'] ='a' 

dbptm_df_2 = dbptm_df_2[['protein', 'organism', 'acc_id', 'ub_mod_loc', 'seq']]
dbptm_df_2['db'] = 'dbptm'
psp_df_2 = psp_df[['protein','organism','acc_id', 'ub_mod_loc',"seq"]]
psp_df_2['db'] = 'psp'
plmd_df_2 = plmd_df[['protein','organism', 'acc_id', 'ub_mod_loc', "seq"]]  
plmd_df_2['db'] = 'plmd'
# %%

ub_df_2 = pd.concat([dbptm_df_2, psp_df_2])

# %%
ub_df_sort = ub_df_2.sort_values(['acc_id', "ub_mod_loc"], ascending=[True,True])
# %%
ub_df_sort.to_csv(folder_path + '/data/ub_sort_two_info.txt', sep='\t', index=False)

# %%

psp = pd.read_csv(folder_path + '/data/psp_info_ord.tsv',
                        sep="\t", low_memory=False, header=None, skiprows=[0, 1])

psp_id = data_psp_valid_entry['acc_id']
# %%
import requests
pred_folder = folder_path + "alpha_pred/"


psp_id = list(set(data_psp_valid_entry['acc_id']))
alphafold_api_url = "https://alphafold.ebi.ac.uk/files/AF-"
af_end_url = "-F1-model_v4.pdb"
links = [f"{alphafold_api_url}{ii}{af_end_url}" for ii in psp_id]
  
with open('af_links.txt', 'w') as tfile:
	tfile.write('\n'.join(links))
#%%
    