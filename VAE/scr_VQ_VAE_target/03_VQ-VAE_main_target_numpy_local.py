#!/usr/bin/env python
# coding: utf-8

# # Extract protein features using VQ-VAE
# ・全細胞のデータを用いるとOS errorが出るので、少しでもメモリを節約するためにLocal変数に書き換える

# In[19]:


import torch
import random
import pickle
import os
import sys
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import random_split

# Import the required files
# sys.path.insert(1, '../download/VAE/codes_target/')
import utils_vq_vae_target, VQ_VAE_target, train_vq_vae_target
from utils_vq_vae_target import *
from VQ_VAE_target import *
from train_vq_vae_target import *

# Seed for reproduction
np.random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# =================================================================================
# Protein signatures.
# =================================================================================
parser = argparse.ArgumentParser()

# Perturbation type.
parser.add_argument(
    '--pert_type', type=str, 
#     default='trt_sh.cgs', 
    default = 'trt_oe',
    help='Perturbation type of protein signatures, e.g., trt_sh.cgs or trt_oe'
)

# Cell line.
parser.add_argument(
    '--cell_name', type=str, 
#     default='All',
#     default='AllCell', 
    default='AllCellParallel',
    help='Cell name, e.g., All: averaged data, AllCell: all signatures measured in all cell lines, AllCellParallel'
)

# Gene scaling.
parser.add_argument(
    '--gene_scaling', type=str, 
    default='MaxAbs', 
#     default='Orig',
    help='Gene scaling, e.g., when not scaling, Orig or when scaling, Std, Cent or MaxAbs'
)

# Select Cell lines based on missing rates.
parser.add_argument(
    '--cell_missing_rate', type=float, 
    default=0.1,
    help='Threshold of missing rates of cell lines when selecting AllCellParallel as --cell_name'
)

# Signature path.
parser.add_argument(
    '--gene_expression_file', type=str, default='../../../LINCS/latest_version/imputation/imputed_data/', 
    help='Path of the training gene expression profile dataset for the VQ-VAE'
)

#===========================================================
# Hyperparameters
#===========================================================
# Training epochs default= 500
parser.add_argument('--gene_epochs', type=int, default=500, help='VQ-VAE training epochs') # Training epochs default= 500

# Length of gene expression signatures.
parser.add_argument('--gene_num', type=int, default=978, help='Number of gene values')

# Lyer size of VQ-VAE encoder (978 -> 1000 -> 512 -> 256)
parser.add_argument('--gene_emb_dim', type=int, 
                    default=[512, 256, 128],
                    # default=[256, 128, 256, 128],  
                    # default=[512, 256, 512, 256], 
                    # default=[512, 1024, 512], 
                    help='Lyer size of VQ-VAE encoder') #  (978 -> 1000 -> 512 -> 256)

# Number of embedding representations (codebook)
parser.add_argument('--gene_num_emb', type=int, 
                    default=64, 
                    # default=128, 
                    # default=256, 
                    help="Number of embedding representations (codebook)") 

# Commitment cost for VQ-VAE
parser.add_argument('--gene_com_cost', type=float, default=0.25, help='Commitment cost for VQ-VAE')

# Batch size
parser.add_argument('--gene_batch_size', type=int, default=64, help='Batch size for training VQ-VAE')

# Learning rate of VQ-VAE model (Default: 2e-3)
parser.add_argument('--gene_lr', type=float, 
#                     default=2e-3,
                    default=2e-4, 
                    help='Learning rate of VQ-VAE, Default: 2e-3')

# Activation function.
parser.add_argument('--gene_activation_fn', type=str, 
                    default='LeakyReLU0.5', 
#                     default='SELU', 
#                         default='Tanh',
                    help='Activation function: Tanh, ReLU or SELU') # Revise: ReLU -> Tanh

# Gene dropout.
parser.add_argument('--gene_dropout', type=float, default=0.1, 
                    help='Dropout probability') # Revise: 0.2 -> 0.1


#===========================================================
# File paths
#===========================================================
parser.add_argument('--saved_gene_vae', type=str, default='saved_vq_vae', help='Save the trained VQ-VAE')

parser.add_argument('--gene_vae_train_results', type=str, default='vq_vae_train_results.csv', 
                    help='Path to save the results of trained VQ-VAE')

parser.add_argument('--one_gene_density_figure', type=str, default='one_gene_density_figure.png', 
                    help='Path to save the density figures of gene data')

parser.add_argument('--all_gene_density_figure', type=str, default='all_gene_density_figure.png', 
                    help='Path to save the density figures of gene data')

args = parser.parse_args()
# args = parser.parse_args(args=[])


#===========================================================
# Print GeneVAE hyperparameter information
#===========================================================
show_gene_vae_hyperparamaters(args)


# In[20]:


# Train GeneVAE for representation learning of gene expression profiles
trained_vq_vae = train_vq_vae(args)
# Note that the VQ-VAE was trained 10 epochs (default: 500 epochs) 


# In[21]:


# ====================================================
# Load trained VQ-VAE model
# ====================================================

gene_activation_fn = get_activation_fn(args)

#  Define the trained GeneVAE.  
if args.cell_name != 'AllCellParallel':
    trained_vq_vae = VQ_VAE(
        gene_num=args.gene_num,
        num_emb=args.gene_num_emb,  
        emb_dim=args.gene_emb_dim,   
        com_cost=args.gene_com_cost,
        activation_fn=gene_activation_fn,
        dropout=args.gene_dropout
    ).to(get_device())
    
elif args.cell_name == 'AllCellParallel':
    cell_lines = select_clls_based_on_missing_rates(args)
    trained_vq_vae = VQ_VAE_MultiCells(
        gene_num=args.gene_num,
        num_emb=args.gene_num_emb,  
        emb_dim=args.gene_emb_dim,   
        com_cost=args.gene_com_cost,
        activation_fn=gene_activation_fn,
        dropout=args.gene_dropout,
        num_cell_lines=len(cell_lines)
    ).to(get_device())

# Load the trained GeneVAE
trained_vq_vae.load_model( make_output_directory_path(args) + args.saved_gene_vae + '.pkl') 
print('Load the trained VQ-VAE.')
print(make_output_directory_path(args) + args.saved_gene_vae + '.pkl')

# Test GeneVAE 
if args.cell_name != 'AllCellParallel':
    show_all_gene_densities( args, trained_vq_vae)
elif args.cell_name == 'AllCellParallel':
    show_all_gene_densities_MultiCell( args, trained_vq_vae )
print('Gene expression profile distribution is created.')


# In[22]:


#==================================================================
# Loss distribution.
#==================================================================

loss_df = pd.read_csv(f"{ make_output_directory_path(args) }/{args.gene_vae_train_results}", sep=',')

sns.relplot(x='Epoch', y='Total', data=loss_df, kind='line')
plt.savefig( f"{ make_output_directory_path(args) }/Total_loss.png", dpi=150 )
sns.relplot(x='Epoch', y='Rec', data=loss_df, kind='line')
plt.savefig( f"{ make_output_directory_path(args) }/Rec_loss.png", dpi=150 )
sns.relplot(x='Epoch', y='Vq', data=loss_df, kind='line')
plt.savefig( f"{ make_output_directory_path(args) }/Vq_loss.png", dpi=150 )
# plt.show()


# In[23]:


#==================================================================
# Extract features from gene expression profiles
#==================================================================
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
trained_vq_vae.eval()
# train_loader = load_gene_expression_dataset(args) # Note that "Shuffle=True"

# Data loader
if args.cell_name != 'AllCellParallel':
    train_loader = load_gene_expression_dataset(args)
    print("Complete dataload")
    
    for (_, genes) in enumerate(train_loader):
        genes = genes.to(get_device())
        print(genes.shape)

        gene_latent_vectors = trained_vq_vae.encoder(genes).to(get_device()) # [batch_size, gene_latent_size]
        # Operate on a batch of gene expression features
        print(gene_latent_vectors.size())
        print(gene_latent_vectors)
        break
    
    
elif args.cell_name == 'AllCellParallel':
    train_loader, num_cell_lines = load_gene_expression_dataset_MultiCell(args)  
#     dropout_fn = nn.Dropout(p=args.gene_dropout)
    print("Complete dataload")
    
    for (_, genes) in enumerate(train_loader):
        genes = genes.to(get_device())
        print(genes.shape)
        
        # Each cell line encoders.
        each_cell_out = []
        for idx, enc in enumerate(trained_vq_vae.each_cell_encoder):
            each_cell_out.append(enc(genes[:,idx]))
        each_cell_out = torch.cat(each_cell_out, dim=1).to(get_device()) # [batch_size, gene_latent_size]
        gene_latent_vectors = trained_vq_vae.integrated_encoder(each_cell_out)
        # Operate on a batch of gene expression features
        print(gene_latent_vectors.size())
        print(gene_latent_vectors)
        break

#         # Each cell line encoders.
#         each_cell_out = []
#         for idx, enc in enumerate(trained_vq_vae.each_cell_encoder):
#             out = enc(genes[:,idx])
#             out = gene_activation_fn(out)
#             out = dropout_fn(out)
#             each_cell_out.append(out)
#         each_cell_out = torch.cat(each_cell_out, dim=1).to(get_device()) # [batch_size, gene_latent_size]
#         gene_latent_vectors = trained_vq_vae.integrated_encoder(each_cell_out)
#         # Operate on a batch of gene expression features
#         print(gene_latent_vectors.size())
#         print(gene_latent_vectors)
#         break


# In[24]:


# ========================================================
# Original gene expression signatures for each cell line
# ========================================================

"""
選択した細胞の情報のみ使う場合
"""

if args.cell_name != 'AllCellParallel':

    gene_data = pd.read_csv(args.gene_expression_file + args.pert_type + '.txt', sep='\t')

    # ----- Mean target signatures across cell lines ----- #
    if args.cell_name == "All":
        gene_data = gene_data.drop('cell_mfc_name', axis = 1
                                  ).groupby(by = 'cmap_name').mean().reset_index() # Average values for each protein
    elif args.cell_name == "AllCell":
        gene_data = gene_data.drop('cell_mfc_name', axis = 1) # Use all cell lines' signatures
    else:
        gene_data = gene_data[ gene_data['cell_mfc_name'] == args.cell_name ].drop('cell_mfc_name', axis = 1) # Select cell line.

    gene_data = gene_data.set_index( 'cmap_name' ) # Set index.


    # ------ Normalize data per gene ------ #
    if args.gene_scaling == 'Orig':
        gene_data = gene_data # Original data
    elif args.gene_scaling == 'Std': 
        gene_data = (gene_data - gene_data.mean())/gene_data.std()
    elif args.gene_scaling == 'Cent': 
        gene_data = (gene_data - gene_data.mean())
    elif args.gene_scaling == 'MaxAbs':
        transformer = MaxAbsScaler() # Define transformer.
        X_scaled = transformer.fit_transform(gene_data) # MaxAbsScaler
        X_scaled = pd.DataFrame(X_scaled, index=gene_data.index, columns=gene_data.columns) # Numpy -> Pandas
        gene_data = X_scaled.replace(0, np.nan
                                ).dropna(how = 'all', axis = 1).fillna(0) # Remove genes with 0 values for all samples
        del X_scaled

    all_gene_list = sorted(set(gene_data.index)) # All gene list

    print('Gene data: ', gene_data.shape)
    print(gene_data)


# In[25]:


# ========================================================
# Extract feature vectors for each cell line
# ========================================================

"""
選択した細胞の情報のみ使う場合
"""

if args.cell_name != 'AllCellParallel':

    # Tensor data type <- pandas 
    gene_data_tensor = gene_data.values.astype('float32') # Pandas -> numpy
    gene_data_tensor = torch.tensor( gene_data_tensor ).float()
    print("Size of the original gene expression signatures: {}".format(gene_data_tensor.shape) )

    # Fix seed values for reproducibility.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Extract latent vectors of gene exppression signatures.
    trained_vq_vae.eval() # 推論モードに切り替える
    feature_data = trained_vq_vae.encoder( gene_data_tensor ).to(get_device()) # Extract latent and output features (0: latent features, 1: output features)
    feature_data = feature_data.to(get_device()).detach().numpy().copy() # tensor -> numpy array
    feature_data = pd.DataFrame( feature_data, index=gene_data.index )
    feature_data.to_csv( make_output_directory_path(args) + 'latent_vectors.txt', sep = '\t')

    print("Size of the embedded gene expression signatures: {}".format(feature_data.shape))

    print(feature_data)


# In[26]:


# ========================================================
# Original gene expression signatures for all cell lines
# ========================================================

"""
全細胞の情報を並列に計算する場合
"""

if args.cell_name == 'AllCellParallel':

    # ===== Load gene expression data ===== #
    gene_data = pd.read_csv(args.gene_expression_file + args.pert_type + '.txt', sep='\t')
    all_gene_list = sorted(set(gene_data['cmap_name'])) # All gene list


    # ===== Missing rate data ====== #
    i_f = "../../../target_repositioning2/multitask/data/LINCS/cell_kdoe_list.txt"
    miss_df = pd.read_csv( i_f, sep ='\t', encoding='shift-jis' )

    if args.pert_type == 'trt_sh.cgs':
        col_name = 'kd_missing'
    elif args.pert_type == 'trt_oe':
        col_name = 'oe_missing'

    # Select cell lines based on missing rate.
    miss_df = miss_df.dropna(subset=col_name) # Select overexpression signature's cells
    miss_df = miss_df[miss_df[col_name] >= args.cell_missing_rate ] # Select cell lines based on missing rate
    cell_lines = sorted(set(miss_df['cell'])) # Cell line list.
    print("Selected cell lines: {}".format(len(cell_lines)) )


    # ===== Normalize for each cell line ====== #
    processed_gene_data = pd.DataFrame()
    for cell in cell_lines:

        # ----- Select cell line. ------ #
        cell_gene_data = gene_data[gene_data['cell_mfc_name'] == cell ].drop('cell_mfc_name', axis = 1) # Select cell line.
        cell_gene_data = cell_gene_data.set_index( 'cmap_name' ) # Set index.
        cell_gene_data = cell_gene_data.loc[all_gene_list] # 全細胞でタンパクの並び順を揃える

        # ------ Normalize data per gene ------ #
        if args.gene_scaling == 'Orig':
            cell_gene_data = cell_gene_data # Original data
        elif args.gene_scaling == 'Std': 
            cell_gene_data = (cell_gene_data - cell_gene_data.mean())/cell_gene_data.std()
        elif args.gene_scaling == 'Cent': 
            cell_gene_data = (cell_gene_data - cell_gene_data.mean())
        elif args.gene_scaling == 'MaxAbs':
            transformer = MaxAbsScaler() # Define transformer.
            X_scaled = transformer.fit_transform(cell_gene_data) # MaxAbsScaler
            X_scaled = pd.DataFrame(X_scaled, index=cell_gene_data.index, columns=cell_gene_data.columns) # Numpy -> Pandas
            cell_gene_data = X_scaled.replace(0, np.nan
                                    ).dropna(how = 'all', axis = 1).fillna(0) # Remove genes with 0 values for all samples
            del X_scaled

        cell_gene_data.insert(0, 'cell_mfc_name', cell) # cell name
        cell_gene_data = cell_gene_data.reset_index() # reset index
        processed_gene_data = pd.concat([processed_gene_data, cell_gene_data], axis = 0)


    # ===== 各タンパク質いついて、全細胞の情報をマージしたnumpy arrayを作成する ===== #

    processed_gene_array = []

    for protein in all_gene_list:

        cell_inputs = processed_gene_data[processed_gene_data['cmap_name'] == protein] # Select protein rows (cell lines x 978 genes).
        cell_inputs = cell_inputs.sort_values(by = 'cell_mfc_name') # Sort cell line names.
        cell_inputs = cell_inputs.drop(['cmap_name', 'cell_mfc_name'], axis = 1 ) # Drop columns.
        processed_gene_array = processed_gene_array + [cell_inputs.values] # Add input data

    else:
        processed_gene_array = np.array(processed_gene_array).astype('float32') # list => numpy array

    print(processed_gene_array.shape)


# In[27]:


# ====== Extract feature vectors ====== #
    
if args.cell_name == 'AllCellParallel':

    # Tensor data type <- pandas 
    # gene_data_tensor = gene_data.values.astype('float32') # Pandas -> numpy
    gene_data_tensor = torch.tensor( processed_gene_array ).float()
    print("Size of the original gene expression signatures: {}".format(gene_data_tensor.shape) )

    # Fix seed values for reproducibility.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Extract latent vectors of gene exppression signatures.
    trained_vq_vae.eval() # 推論モードに切り替える
    dropout_fn = nn.Dropout(p=args.gene_dropout)

    # Each cell line encoders.
    each_cell_out = []
    for idx, enc in enumerate(trained_vq_vae.each_cell_encoder):
        each_cell_out.append(enc(gene_data_tensor[:,idx]))
    #     out = enc(gene_data_tensor[:,idx])
    #     out = gene_activation_fn(out)
    #     out = dropout_fn(out)
    #     each_cell_out.append(out)
    each_cell_out = torch.cat(each_cell_out, dim=1).to(get_device()) # [batch_size, gene_latent_size]
    feature_data = trained_vq_vae.integrated_encoder(each_cell_out) # Extract latent and output features (0: latent features, 1: output features)
    feature_data = feature_data.to(get_device()).detach().numpy().copy() # tensor -> numpy array
    feature_data = pd.DataFrame( feature_data, index=all_gene_list )
    feature_data.to_csv( make_output_directory_path(args) + 'latent_vectors.txt', sep = '\t')

    print("Size of the embedded gene expression signatures: {}".format(feature_data.shape))

    print(feature_data)


# In[ ]:





# In[10]:


# #==================================================================
# # Read the data, which contains smiles, inchikey, and gene values
# #==================================================================

# gene_data = pd.read_csv(args.gene_expression_file + args.pert_type + '.txt', sep='\t')

# # ------ Mean target signatures across cell lines. ------ #
# if args.cell_name == "All":
#     gene_data = gene_data.drop('cell_mfc_name', axis = 1
#                               ).groupby(by = 'cmap_name').mean().reset_index() # Average values for each protein
# elif args.cell_name == "AllCell":
#     gene_data = gene_data.drop('cell_mfc_name', axis = 1) # Use all cell lines' signatures
# else:
#     gene_data = gene_data[ gene_data['cell_mfc_name'] == args.cell_name ].drop('cell_mfc_name', axis = 1) # Select cell line.
    
# gene_data = gene_data.set_index( 'cmap_name' ) # Set index.


# # ------ Normalize data per gene ------ #
# if GENE_SCALING == 'Orig':
#     gene_data = gene_data # Original data
# elif GENE_SCALING == 'Std': 
#     gene_data = (gene_data - gene_data.mean())/gene_data.std()
# elif GENE_SCALING == 'Cent': 
#     gene_data = (gene_data - gene_data.mean())
# elif GENE_SCALING == 'MaxAbs':
#     transformer = MaxAbsScaler() # Define transformer.
#     X_scaled = transformer.fit_transform(gene_data) # MaxAbsScaler
#     X_scaled = pd.DataFrame(X_scaled, index=gene_data.index, columns=gene_data.columns) # Numpy -> Pandas
#     gene_data = X_scaled.replace(0, np.nan
#                             ).dropna(how = 'all', axis = 1).fillna(0) # Remove genes with 0 values for all samples
#     del X_scaled
    
    
# gene_data_index = list(gene_data.index)

# gene_data = gene_data.values.astype('float32') # Pandas -> numpy
    

# print('Gene data: ', gene_data.shape)
# gene_data


# In[11]:


# # ====== Extract feature vectors ====== #

# # Tensor data type <- pandas 
# # gene_data_tensor = gene_data.values.astype('float32') # Pandas -> numpy
# gene_data_tensor = torch.tensor( gene_data_tensor ).float()
# print("Size of the original gene expression signatures: {}".format(gene_data_tensor.shape) )

# # Fix seed values for reproducibility.
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)

# # Extract latent vectors of gene exppression signatures.
# trained_vq_vae.eval() # 推論モードに切り替える
# feature_data = trained_vq_vae.encoder( gene_data_tensor ).to(get_device()) # Extract latent and output features (0: latent features, 1: output features)
# feature_data = feature_data.to(get_device()).detach().numpy().copy() # tensor -> numpy array
# feature_data = pd.DataFrame( feature_data, index=gene_data.index )
# feature_data.to_csv( make_output_directory_path(args) + 'latent_vectors.txt', sep = '\t')

# print("Size of the embedded gene expression signatures: {}".format(feature_data.shape))

# feature_data

