import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import argparse
import os
# from rdkit import Chem
# from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
# from rdkit.DataStructs import FingerprintSimilarity

# ============================================================================
# KL Divergence loss 
def kld_loss(mu, logvar):
    """
    mu: Means of encoder output [batch_size, latent_size]
    logvar: log varances of encoder output [batch_size, latent_size]
    returns:
        KLD of the specified distribution and a unit Gaussian.
    """

    mu = mu.double().to(get_device())
    logvar = logvar.double().to(get_device())

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return kld

# ============================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
def make_output_directory_path(args):

    # Meke output data directory
    o_dir = f"../data/VAE/{args.pert_type}/CellLine_{args.cell_name}" # Cell line
    o_dir = f"{o_dir}/Epo{args.gene_epochs}_Lr{args.gene_lr}" # Epochs
    o_dir = f"{o_dir}_Hid{'_'.join([str(s) for s in sorted(args.gene_hidden_sizes, reverse = True)])}" # hidden sizes
    o_dir = f"{o_dir}_Lat{args.gene_latent_size}" # latent size
    o_dir = f"{o_dir}_Bat{args.gene_batch_size}" # batch size
    o_dir = f"{o_dir}_Dro{args.gene_dropout}" # drop out
    o_dir = f"{o_dir}_Act{args.gene_activation_fn}/" # activation function
    os.makedirs( o_dir, exist_ok = True ) # make directory
    return o_dir

# ============================================================================
def show_gene_vae_hyperparamaters(args):

    # Hyper-parameters
    params = {}
    print('\n\nGeneVAE Hyperparameter Information:')
    print('='*50)
    params['PERTURBATION_TYPE'] = args.pert_type 
    params['GENE_EXPRESSION_FILE'] = args.gene_expression_file 
    params['CELL_NAME'] = args.cell_name
    params['GENE_EPOCHS'] = args.gene_epochs 
    params['GENE_LR'] = args.gene_lr
    params['GENE_NUM'] = args.gene_num
    params['GENE_HIDDEN_SIZES'] = args.gene_hidden_sizes
    params['GENE_LATENT_SIZE'] = args.gene_latent_size
    params['GENE_BATCH_SIZE'] = args.gene_batch_size
    params['GENE_DROUPOUT'] = args.gene_dropout
    params['GENE_ACTIVATION_FUNCTION'] = args.gene_activation_fn

    for param in params:
        string = param + ' ' * (5 - len(param))
        print('{}:   {}'.format(string, params[param]))
    print('='*50)

# ============================================================================
# Build vocabulary for SMILES data 
class Tokenizer():

    def __init__(self):
        self.start = "^"
        self.end = "$"
        self.pad = ' '
    
    def build_vocab(self):
        chars=[]
        # atoms 
        chars = chars + ['H', 'B', 'C', 'c', 'N', 'n', 'O', 'o', 'P', 'S', 's', 'F', 'I']
        # replace Si for Q, Cl for R, Br for V
        chars = chars + ['Q', 'R', 'V', 'Y', 'Z', 'G', 'T', 'U']
        # hidrogens: H2 to W, H3 to X
        chars = chars + ['[', ']', '+', 'W', 'X']
        # bounding
        chars = chars + ['-', '=', '#', '.', '/', '@', '\\']
        # branches
        chars = chars + ['(', ')']
        # cycles
        chars = chars + ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        #padding value is 0
        self.tokenlist = [self.pad, self.start, self.end] + list(chars)
        # create the dictionaries      
        self.char_to_int = {c:i for i,c in enumerate(self.tokenlist)}
        self.int_to_char = {i:c for c,i in self.char_to_int.items()}
    
    @property
    def vocab_size(self):
        return len(self.int_to_char)
    
    def encode(self, smi):
        encoded = []
        smi = smi.replace('Si', 'Q')
        smi = smi.replace('Cl', 'R')
        smi = smi.replace('Br', 'V')
        smi = smi.replace('Pt', 'Y')
        smi = smi.replace('Se', 'Z')
        smi = smi.replace('Li', 'T')
        smi = smi.replace('As', 'U')
        smi = smi.replace('Hg', 'G')
        # hydrogens
        smi = smi.replace('H2', 'W')
        smi = smi.replace('H3', 'X')

        return [self.char_to_int[self.start]] + [self.char_to_int[s] for s in smi] + [self.char_to_int[self.end]]
    
    def decode(self, ords):
        smi = ''.join([self.int_to_char[o] for o in ords]) 
        # hydrogens
        smi = smi.replace('W', 'H2')
        smi = smi.replace('X', 'H3')
        # replace proxy atoms for double char atoms symbols
        smi = smi.replace('Q', 'Si')
        smi = smi.replace('R', 'Cl')
        smi = smi.replace('V', 'Br')
        smi = smi.replace('Y', 'Pt')
        smi = smi.replace('Z', 'Se')
        smi = smi.replace('T', 'Li')
        smi = smi.replace('U', 'As')
        smi = smi.replace('G', 'Hg')
        
        return smi

    @property
    def n_tokens(self):
        return len(self.int_to_char)

# ============================================================================
def vocabulary(args):

    # Build the vocabulary
    tokenizer = Tokenizer()
    tokenizer.build_vocab()
    #print('\n')
    #print('Vocabulary Information:')
    #print('='*50)
    #print(tokenizer.char_to_int)
    #print('='*50)

    return tokenizer

# ============================================================================
def show_density(
    args, 
    figure_path, 
    row_num, 
    trained_gene_vae=None,
    gene_idx=None
):
    """
    figure_path: the path to save the figure
    row_num: number of rows of gene expression profile data used for data distribution
    """

    # Real gene expression profile data loading
    real_genes = pd.read_csv(
        args.gene_expression_file + args.pert_type +'.txt', 
        sep='\t' 
    )

    # Mean target signatures across cell lines
    if args.cell_name == "All":
        real_genes = real_genes.drop('cell_mfc_name', axis = 1).groupby(by = 'cmap_name').mean().reset_index() # Average gene expression signatures across all cell lines.
    elif args.cell_name == "All_AllCell":
        averaged_data = real_genes.drop('cell_mfc_name', axis = 1).groupby(by = 'cmap_name').mean().reset_index() # 平均
        averaged_data.insert( 1, 'cell_mfc_name', 'All' ) # add cell line columns

        real_genes = pd.concat([averaged_data, real_genes], axis=0) # Concatenate averaged data and original data.
        real_genes['cmap_name'] = [ f"{a}@{b}" for a,b in zip(real_genes['cmap_name'], real_genes['cell_mfc_name']) ] # Merge protein names and cell lines
        real_genes = real_genes.drop('cell_mfc_name', axis = 1)
    else:
        real_genes = real_genes[ real_genes['cell_mfc_name'] == args.cell_name].drop('cell_mfc_name', axis = 1) # Select cell line.
    real_genes = real_genes.set_index( 'cmap_name' ) # indexを指定

    # Drop the nan row
    real_genes = real_genes.dropna(how='any')
    # Normalize data per gene
    #real_genes = (real_genes - real_genes.mean())/real_genes.std()

    # Calculate average value
    if row_num == 1:
        random_rows = np.array([gene_idx])
        #random_rows = np.random.choice(len(real_genes), row_num)
    else:
        random_rows = np.random.choice(len(real_genes), row_num)
    real_genes = real_genes.iloc[random_rows, :]
    mean_real_all_gene = real_genes.mean()

    plt.subplots(figsize=(12,7))
    plt.title("Data distribution of gene expression profile", fontsize=12)
    plt.xlabel("Values of gene expression profile data", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    # Figure density distribution
    sns.histplot(mean_real_all_gene, bins=50, kde=True, label='Real gene', color='g')
    
    if trained_gene_vae:
        trained_gene_vae.eval()
        # Reconstructed gene
        inputs = torch.tensor(real_genes.values, dtype=torch.float32).to(get_device())
        _, rec_genes = trained_gene_vae(inputs)
        rec_genes = pd.DataFrame(rec_genes.cpu().detach().numpy())
        # Calculate average value
        mean_rec_gene = rec_genes.mean()
        # Figure density distribution
        sns.histplot(mean_rec_gene, bins=50, kde=True, label='Reconstructed gene', color='r')
    
    plt.legend()
    plt.savefig(figure_path, dpi=150)

def show_all_gene_densities(args, trained_gene_vae):

    show_density(args, f"{make_output_directory_path(args)}/{args.all_gene_density_figure}", 10000, trained_gene_vae)
    
    for gene_idx in range(1,11):
        show_density(args, f"{make_output_directory_path(args)}/one_gene_density_figure_{gene_idx}.png", 1, trained_gene_vae, gene_idx)
        # show_density(args, f"{make_output_directory_path(args)}/{args.one_gene_density_figure}", 1, trained_gene_vae, gene_idx=gene_idx)

# ============================================================================
def symbol2hsa(input_symbol):
    with open('datasets/tools/symbol2hsa.json', mode='rt', encoding='utf-8')as f:
        symbol_data = json.load(f)
        symbols = list(symbol_data.keys())
    hsas = []
    for sym in input_symbol:
        if sym in symbols:
            hsas.append(symbol_data[sym])
        else:
            hsas.append('-')
    return hsas

def common(df_tgt, gene_type):
    # Source gene names
    df_source = pd.read_csv('datasets/tools/source_genes.csv', sep=',')
    source_hsas = list(df_source.columns)
    # Target gene names
    tgt_hsas = list(df_tgt.columns)
    
    if not gene_type == 'gene_symbol':
        tgt_hsas = symbol2hsa(tgt_hsas)
        df_tgt = df_tgt.set_axis(tgt_hsas, axis=1)
   
    # Common gene names
    common_hsas = list(set(tgt_hsas) & set(source_hsas))
    common_hsas = sorted(common_hsas, key=source_hsas.index)
    # Processed target gene expression profile data
    df_source[common_hsas] = df_tgt[common_hsas]
    
    return df_source












