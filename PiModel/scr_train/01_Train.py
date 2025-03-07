#!/usr/bin/env python
# coding: utf-8

# # Pi model for predicting therapeutic targets 

# In[1]:


import sys  
import torch
import random
import argparse
import numpy as np
import time
import torch.nn as nn
import pandas as pd

# Apply the seed to reproduct the results
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# =================================================================================
# Protein signatures
# =================================================================================
parser = argparse.ArgumentParser()

# Perturbation type.
parser.add_argument('--pert_type', type=str, default='trt_sh.cgs', help='Perturbation type of protein signatures, e.g., trt_sh.cgs or trt_oe') # knockdown signatures
# parser.add_argument('--pert_type', type=str, default='trt_oe', help='Perturbation type of protein signatures, e.g., trt_sh.cgs or trt_oe') # overexpression signatures

# Cell line.
parser.add_argument('--cell_name', type=str, default='AllCellParallel', help='Cell name of LINCS files, e.g., mcf7') # cell line of protein signatures

# Tagets used for training.
parser.add_argument('--target_select', type=str, default='Known', 
                    help='Targets used for training. If supervised learning, only known proteins are used for training.')

# Target feature type.
parser.add_argument('--target_feature_type', type=str, default='Epo500_Lr0.0002_Hid512_1024_512_Lat256_Bat64_Comc0.25_Dro0.1_ActLeakyReLU0.5_Missing0.1_ScaleMaxAbs', 
                    help='target feature types, e.g., Epo500_Lr0.002_Hid1000_512_256_Lat128_Bat64_Comc0.25, VarianceTop1 or Original')

# Target expression profile.
parser.add_argument('--target_expression_file', type=str, default='../../../LINCS/latest_version/imputation/imputed_data/', 
                    help='Path of the training gene expression profile dataset for the NN')

# Target vae type.
parser.add_argument('--target_vae_type', type=str, default='VQ_VAE', help='Protein VAE type, e.g., VAE, VQ_VAE or Original')

# =================================================================================
# Disease signatures
# =================================================================================
parser.add_argument('--disease_profile_type', type=str, default='disease_signature.tabs', 
                    help='Disease signature type, e.g., disease_signature.tabs or patient_signature.tabs') # disease signatures

# Diseases used for training.
parser.add_argument('--disease_select', type=str, default='79', 
                    help='Diseases used for training. If supervised learning, only known diseases are used for training.')

# Disease feature type.
parser.add_argument('--disease_feature_type', type=str, default='Epo2000_Lr0.002_Hid1000_512_256_Lat128_Bat64_Comc0.25_ScaleStd', 
                    help='disease feature types, e.g., Epo2000_Lr0.0001_Hid1024_512_256_Lat128_Bat64_Dro0.2_ScaleMaxAbs_ActReLU, VarianceTop1 or Original')

# Disease expression profile.
parser.add_argument('--disease_expression_file', type=str, default='../../../CREEDS/processed_data/expression-based_', 
                    help='Path of the training gene expression profile dataset for the VAE')

# Disease vae type.
parser.add_argument('--disease_vae_type', type=str, default='VQ_VAE', help='Disease VAE type, e.g., VAE, VQ_VAE or Original')

# =================================================================================
# Pairwise type.
# =================================================================================
parser.add_argument('--pairwise_type', type=str, default='Concat', 
                    help='Feature types of protein–disease pairs, e.g., Concat or DisCosine_TarCosine_Kron')


# =================================================================================
# Default settings
# =================================================================================
parser.add_argument('--fold_number', type=str, default="1", 
                    help='Cross validation fold number, e.g., if fold number is 1, else if newprediction, newprediction')
parser.add_argument('--use_seed', action='store_true', help='Apply seed for reproduce experimental results') # Add --use_seed for reproduction


# =================================================================================
# Hyper-parameters of GeneVAE
# =================================================================================
# Epochs
parser.add_argument('--gene_epochs', type=int, default=10, help='NN training epochs') # default: 2000
# parser.add_argument('--gene_num', type=int, default=15782, help='Number of gene values of protein and disease signatures, e.g., original=>15782')

# Hidden layer sizes
# parser.add_argument('--gene_hidden_sizes', type=int, default=[1024, 512, 256], help='Hidden layer sizes of NN')
parser.add_argument('--gene_hidden_sizes', type=int, default=[512, 256, 128], nargs="*", 
                    help='Hidden layer sizes of NN')

# Gene output size
parser.add_argument('--gene_output_size', type=int, default=1, help='Output vector dimension of NN') 

# Learning rate
parser.add_argument('--gene_lr', type=float, default=1e-4, help='Learning rate of NN') 

# Optimizer
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer: AdamW or SGD')
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--weight_decay', type=float, default=5e-4)

# Batch size
parser.add_argument('--gene_batch_size', type=int, default=64, help='Batch size for training NN') 

# Dropout
parser.add_argument('--gene_dropout', type=float, default=0.2, help='Dropout probability')

# Loss function
parser.add_argument('--loss_fn', type=str, default="CELweight", help='Loss function, e.g., CEL, CELweight, and FL')

# Early stopping
parser.add_argument('--early_stopping', type=str, default="30", help='Early stopping patience, if not early stopping, No')

# EMA
parser.add_argument('--ema_m', type=float, default=0.999)

# Max unsupervised weight.
parser.add_argument('--ulb_loss_ratio', type=float, default=10.0)

# Unsupervised loss warmup ends.
parser.add_argument('--unsup_warmup_pos', type=float, default=0.4,
                        help='Relative position at which constraint loss warmup ends.')

# Total number of training iterations
parser.add_argument('--num_train_iter', type=int, default=64 * 2000,
                        help='total number of training iterations')


# =================================================================================
# File paths.
# =================================================================================
parser.add_argument('--saved_gene_nn', type=str, default='saved_gene_nn', help='Save the trained Pi model')
parser.add_argument('--gene_nn_train_results', type=str, default='gene_nn_train_results.csv', help='Path to save the results of trained Pi model')
args = parser.parse_args()
# args = parser.parse_args(args=[])

# =================================================================================
# Import the required files
# =================================================================================
# sys.path.insert(1, '../scr_train/')
# if args.early_stopping == 'No':
#     import utils_nn, ConcatNN, train_gene_nn
#     from utils_nn import *
#     from ConcatNN import *
#     from train_gene_nn import *
# else:
import utils_pimodel, ConcatNN, train_pimodel_earlystopping
from utils_pimodel import *
from ConcatNN import *
from train_pimodel_earlystopping import *

# ========================================================= #
# Print GeneVAE hyperparameter information
# =================================================================================
show_gene_vae_hyperparamaters(args)


# In[3]:


# if __name__ == “__main__”:


# In[2]:


# Train ConcatNN for representation learning of gene expression profiles

start = time.time()
if args.early_stopping == "No":
    trained_gene_nn = train_gene_nn(args)
else:
    trained_gene_nn = train_pi_model_earlystopping(args)
# Note that the ConcatNN was trained 10 epochs (default: 2000 epochs) 

# かかった時間
print("=======かかった時間========")
print(time.time() - start)

