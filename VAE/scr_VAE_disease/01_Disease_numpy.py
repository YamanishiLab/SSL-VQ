#!/usr/bin/env python
# coding: utf-8

# # Extract disease features using VAE.

# In[1]:


import sys  
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler

# Import the required files
sys.path.insert(1, '../scr_VAE_disease/')
import utils_disease, GeneVAE_disease, train_gene_vae_disease_numpy
from utils_disease import *
from GeneVAE_disease import *
from train_gene_vae_disease_numpy import *

# Apply the seed to reproduct the results
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# =================================================================================
# Default settings
parser = argparse.ArgumentParser()
parser.add_argument('--profile_type', type=str, default='disease_signature.tabs', 
                    help='Disease signature type, e.g., disease_signature.tabs, patient_signature.tabs or disease_and_patient_signature.tabs')
# parser.add_argument('--profile_type', type=str, default='patient_signature.tabs', help='Disease signature type, e.g., disease_signature.tabs or patient_signature.tabs')
parser.add_argument('--use_seed', action='store_true', help='Apply seed for reproduce experimental results') # Add --use_seed for reproduction
parser.add_argument('--gene_scaling', type=str, default='MaxAbs', help='Gene scaling, e.g., when not scaling, Orig or when scaling, Std or MaxAbs')

# =================================================================================
# Hyper-parameters of GeneVAE

parser.add_argument('--gene_epochs', type=int, default=2000, help='GeneVAE training epochs') # default: 2000

# parser.add_argument('--gene_hidden_sizes', type=int, default=[512, 256, 128], help='Hidden layer sizes of GeneVAE')
parser.add_argument('--gene_hidden_sizes', type=int, default=[1024, 512, 256], help='Hidden layer sizes of GeneVAE')
# parser.add_argument('--gene_hidden_sizes', type=int, default=[2048, 1024, 512], help='Hidden layer sizes of GeneVAE')

# parser.add_argument('--gene_latent_size', type=int, default=64, help='Latent vector dimension of GeneVAE') 
parser.add_argument('--gene_latent_size', type=int, default=128, help='Latent vector dimension of GeneVAE') 
# parser.add_argument('--gene_latent_size', type=int, default=256, help='Latent vector dimension of GeneVAE') 

parser.add_argument('--gene_lr', type=float, default=1e-4, help='Learning rate of GeneVAE') 
parser.add_argument('--gene_batch_size', type=int, default=64, help='Batch size for training GeneVAE') 
parser.add_argument('--gene_dropout', type=float, default=0.2, help='Dropout probability') # Revise: 0.2 -> 0.1
parser.add_argument('--gene_activation_fn', type=str, default='ReLU', help='Activation function: Tanh or ReLU') # Revise: ReLU -> Tanh

# File paths
parser.add_argument('--gene_expression_file', type=str, default='../../../CREEDS/processed_data/expression-based_', 
                    help='Path of the training gene expression profile dataset for the VAE')
parser.add_argument('--saved_gene_vae', type=str, default='saved_gene_vae', help='Save the trained GeneVAE')
parser.add_argument('--gene_vae_train_results', type=str, default='gene_vae_train_results.csv', help='Path to save the results of trained GeneVAE')
parser.add_argument('--one_gene_density_figure', type=str, default='one_gene_density_figure.png', help='Path to save the density figures of gene data')
parser.add_argument('--all_gene_density_figure', type=str, default='all_gene_density_figure.png', help='Path to save the density figures of gene data')
parser.add_argument('--gene_type', type=str, default='gene_symbol', help='Gene types')
parser.add_argument("-f", required=False)
args = parser.parse_args()

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
    input_size=gene_num,
    hidden_sizes=args.gene_hidden_sizes,
    latent_size=args.gene_latent_size,
    activation_fn=gene_activation_fn,
    dropout=args.gene_dropout
).to(get_device())
trained_gene_vae = GeneVAE(
    input_size=gene_num, 
    hidden_sizes=args.gene_hidden_sizes,
    latent_size=args.gene_latent_size,
    output_size=gene_num,
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


# Loss distribution.

loss_df = pd.read_csv(f"{ make_output_directory_path(args) }/gene_vae_train_results.csv", sep=',')

sns.relplot(x='Epoch', y='Joint', data=loss_df, kind='line')
plt.savefig( f"{ make_output_directory_path(args) }/Joint_loss.png", dpi=150 )
sns.relplot(x='Epoch', y='Rec', data=loss_df, kind='line')
plt.savefig( f"{ make_output_directory_path(args) }/Rec_loss.png", dpi=150 )
sns.relplot(x='Epoch', y='KLD', data=loss_df, kind='line')
plt.savefig( f"{ make_output_directory_path(args) }/KLD_loss.png", dpi=150 )
plt.show()


# In[23]:


# Extract features from gene expression profiles
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
trained_gene_vae.eval()
train_loader,gene_num = load_gene_expression_dataset(args) # Note that "Shuffle=True"

for (_, genes) in enumerate(train_loader):
    genes = genes.to(get_device())
    print(genes.shape)
    
    gene_latent_vectors, _ = trained_gene_vae(genes) # [batch_size, gene_latent_size]
    # Operate on a batch of gene expression features
    print(gene_latent_vectors.size())
    print(gene_latent_vectors)
    break


# In[2]:


# Read the disease gene expression data, which contains disease ids and gene values

gene_data = pd.read_csv(args.gene_expression_file + args.profile_type, 
                         sep = '\t',
                         index_col='disease'
                        )

#  Normalize data per gene
if args.gene_scaling == 'Orig':
    gene_data = gene_data # Original data
    
elif args.gene_scaling == 'Std': 
    gene_data = (gene_data - gene_data.mean())/gene_data.std() # Diseaseはスパースなデータなので標準化
    gene_data = gene_data.dropna(how = 'any', axis = 1)
    
elif args.gene_scaling == 'MaxAbs':
    transformer = MaxAbsScaler() # Define transformer.
    X_scaled = transformer.fit_transform(gene_data) # MaxAbsScaler
    X_scaled = pd.DataFrame(X_scaled, index=gene_data.index, columns=gene_data.columns) # Numpy -> Pandas
    gene_data = X_scaled.replace(0, np.nan
                                ).dropna(how = 'all', axis = 1).fillna(0) # Remove genes with 0 values for all samples
    del X_scaled

gene_num = gene_data.shape[1] # Number of genes

print('Gene data: ', gene_data.shape) # (nums, dimension)
gene_data


# In[10]:


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

