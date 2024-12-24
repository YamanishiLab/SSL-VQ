#!/usr/bin/env python
# coding: utf-8

# # Extract protein features using VAE

# In[9]:


import sys  
import torch
import random
import argparse
import numpy as np
import torch.nn as nn

# Import the required files
# sys.path.insert(1, '../download/VAE/codes_target/')
import utils_target, GeneVAE_target, train_gene_vae_target_numpy
from utils_target import *
from GeneVAE_target import *
from train_gene_vae_target_numpy import *

# Apply the seed to reproduct the results
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# =================================================================================
# Default settings
parser = argparse.ArgumentParser()
parser.add_argument('--pert_type', type=str, default='trt_sh.cgs', help='Perturbation type of protein signatures, e.g., trt_sh.cgs or trt_oe')
parser.add_argument('--use_seed', action='store_true', help='Apply seed for reproduce experimental results') # Add --use_seed for reproduction
parser.add_argument('--cell_name', type=str, default='All_AllCell', 
                    help="Cell name of LINCS files, e.g., All: averaged signatures, AllCell: all cell lines' signatures")

# Hyper-parameters of GeneVAE
parser.add_argument('--gene_epochs', type=int, default=2, help='GeneVAE training epochs') # default: 2000
parser.add_argument('--gene_num', type=int, default=978, help='Number of gene values')
parser.add_argument('--gene_hidden_sizes', type=int, default=[512, 256, 128], help='Hidden layer sizes of GeneVAE')
parser.add_argument('--gene_latent_size', type=int, default=64, help='Latent vector dimension of GeneVAE') 
parser.add_argument('--gene_lr', type=float, default=1e-4, help='Learning rate of GeneVAE') 
parser.add_argument('--gene_batch_size', type=int, default=64, help='Batch size for training GeneVAE') 
parser.add_argument('--gene_dropout', type=float, default=0.1, help='Dropout probability') # Revise: 0.2 -> 0.1
parser.add_argument('--gene_activation_fn', type=str, default='Tanh', help='Activation function: Tanh or ReLU') # Revise: ReLU -> Tanh

# File paths
parser.add_argument('--gene_expression_file', type=str, default='../../../LINCS/latest_version/imputation/imputed_data/', 
                    help='Path of the training gene expression profile dataset for the VAE')
parser.add_argument('--saved_gene_vae', type=str, default='saved_gene_vae', help='Save the trained GeneVAE')
parser.add_argument('--gene_vae_train_results', type=str, default='gene_vae_train_results.csv', help='Path to save the results of trained GeneVAE')
parser.add_argument('--one_gene_density_figure', type=str, default='one_gene_density_figure.png', help='Path to save the density figures of gene data')
parser.add_argument('--all_gene_density_figure', type=str, default='all_gene_density_figure.png', help='Path to save the density figures of gene data')
args = parser.parse_args(args=[])

# ========================================================= #
# Print GeneVAE hyperparameter information
show_gene_vae_hyperparamaters(args)


# In[4]:


# Train GeneVAE for representation learning of gene expression profiles
trained_gene_vae = train_gene_vae(args)
# Note that the GeneVAE was trained 10 epochs (default: 2000 epochs) 


# In[3]:


#  Define the trained GeneVAE  

# Activation function
if args.gene_activation_fn == 'ReLU':
    gene_activation_fn = nn.ReLU()
elif args.gene_activation_fn == 'Tanh':
    gene_activation_fn = nn.Tanh()

GeneEncoder(
    input_size=args.gene_num, 
    hidden_sizes=args.gene_hidden_sizes,
    latent_size=args.gene_latent_size,
    activation_fn=gene_activation_fn,
    dropout=args.gene_dropout
).to(get_device())

trained_gene_vae = GeneVAE(
    input_size=args.gene_num, 
    hidden_sizes=args.gene_hidden_sizes,
    latent_size=args.gene_latent_size,
    output_size=args.gene_num,
    activation_fn=gene_activation_fn,
    dropout=args.gene_dropout
).to(get_device())

# Load the trained GeneVAE
# trained_gene_vae.load_model(args.saved_gene_vae + '_' + args.cell_name + '.pkl')
trained_gene_vae.load_model( make_output_directory_path(args) + args.saved_gene_vae + '.pkl') 
print('Load the trained GeneVAE.')

# Test GeneVAE 
show_all_gene_densities( args, trained_gene_vae)
print('Gene expression profile distribution is created.')


# In[4]:


# Extract features from gene expression profiles
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
trained_gene_vae.eval()
train_loader = load_gene_expression_dataset(args) # Note that "Shuffle=True"

for (_, genes) in enumerate(train_loader):
    genes = genes.to(get_device())
    print(genes.shape)
    
    gene_latent_vectors, _ = trained_gene_vae(genes) # [batch_size, gene_latent_size]
    # Operate on a batch of gene expression features
    print(gene_latent_vectors.size())
    print(gene_latent_vectors)
    break


# In[26]:


# Read the data, which contains smiles, inchikey, and gene values
data = pd.read_csv(args.gene_expression_file + args.pert_type +'.txt', sep='\t' )

if args.cell_name == "All":
    gene_data = data.drop('cell_mfc_name', axis = 1
                         ).groupby(by = 'cmap_name').mean().reset_index() # 平均
    print("Averaged signatures")
    
elif args.cell_name == "All_AllCell":
    averaged_data = data.drop('cell_mfc_name', axis = 1).groupby(by = 'cmap_name').mean().reset_index() # 平均
    averaged_data.insert( 1, 'cell_mfc_name', 'All' ) # add cell line columns

    gene_data = pd.concat([averaged_data, data], axis=0) # Concatenate averaged data and original data.
    gene_data['cmap_name'] = [ f"{a}@{b}" for a,b in zip(gene_data['cmap_name'], gene_data['cell_mfc_name']) ] # Merge protein names and cell lines
    gene_data = gene_data.drop('cell_mfc_name', axis = 1)
    print("Averaged and original signatures")
    
else:
    gene_data = data[ data['cell_mfc_name'] == args.cell_name ].drop('cell_mfc_name', axis = 1) # cellを選択
    print(args.cell_name)
    
gene_data = gene_data.set_index( 'cmap_name' ) # indexを指定
print('Gene data: ', gene_data.shape) # (nums, dimension)

gene_data


# In[29]:


# ====== Extract feature vectors ====== #

# Tensor data type <- pandas 
gene_data_tensor = torch.tensor( gene_data.values ).float()
print("Size of the original gene expression signatures: {}".format(gene_data_tensor.shape) )

# Fix seed values for reproducibility.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Extract latent vectors of gene exppression signatures.
trained_gene_vae.eval() # 推論モードに切り替える
feature_data = trained_gene_vae( gene_data_tensor ) # Extract latent and output features (0: latent features, 1: output features)
feature_data = feature_data[0].to(get_device()).detach().numpy().copy() # tensor -> numpy array
feature_data = pd.DataFrame( feature_data, index=gene_data.index )
feature_data.to_csv( make_output_directory_path(args) + 'latent_vectors.txt', sep = '\t')

print("Size of the embedded gene expression signatures: {}".format(feature_data.shape))

feature_data

