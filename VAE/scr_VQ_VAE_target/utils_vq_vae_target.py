import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import argparse
import os
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from torch.distributions.normal import Normal

# ============================================================================
# Get device
# ============================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Make output directory and get output directory path
# ============================================================================
def make_output_directory_path(args):

    # Meke output data directory
    o_dir = f"../data/VQ_VAE/{args.pert_type}/CellLine_{args.cell_name}" # Cell line
    o_dir = f"{o_dir}/Epo{args.gene_epochs}_Lr{args.gene_lr}" # Epochs
    o_dir = f"{o_dir}_Hid{'_'.join([str(s) for s in args.gene_emb_dim])}" # hidden sizes
    o_dir = f"{o_dir}_Lat{args.gene_num_emb}" # latent size
    o_dir = f"{o_dir}_Bat{args.gene_batch_size}" # batch size
    o_dir = f"{o_dir}_Comc{args.gene_com_cost}" # commitment cost
    o_dir = f"{o_dir}_Dro{args.gene_dropout}" # drop out
    o_dir = f"{o_dir}_Act{args.gene_activation_fn}" # activation function

    if args.cell_name == 'AllCellParallel' or args.cell_name == 'AllCell':
        o_dir = f"{o_dir}_Missing{args.cell_missing_rate}" # Threshold of missing rate.
        
    o_dir = f"{o_dir}_Scale{args.gene_scaling}/" # Scaling
    os.makedirs( o_dir, exist_ok = True ) # make directory
    return o_dir

# ============================================================================
# Show hyperparameters
# ============================================================================
def show_gene_vae_hyperparamaters(args):

    # Hyper-parameters
    params = {}
    print('\n\nVQ-VAE Hyperparameter Information:')
    print('='*50)
    params['PROFILE_TYPE'] = args.pert_type
    params['GENE_EXPRESSION_FILE'] = args.gene_expression_file
    params['CELL_NAME'] = args.cell_name
    params['GENE_EPOCHS'] = args.gene_epochs
    params['GENE_LR'] = args.gene_lr
    params['GENE_NUM'] = args.gene_num
    params['GENE_EMB_DIM'] = args.gene_emb_dim
    params['GENE_NUM_EMB'] = args.gene_num_emb
    params['GENE_BATCH_SIZE'] = args.gene_batch_size
    params['GENE_DROUPOUT'] = args.gene_dropout
    params['GENE_ACTIVATION_FUNCTION'] = args.gene_activation_fn

    for param in params:
        string = param + ' ' * (5 - len(param))
        print('{}:   {}'.format(string, params[param]))
    print('='*50)


# ============================================================================
# Define gene expression dataset
class GeneExpressionDataset(torch.utils.data.Dataset):
    
    def __init__(self, data):

        self.data = data
        self.data_num = len(data)

    def __len__(self):

        return self.data_num

    def __getitem__(self, idx):
        gene_data = torch.tensor(self.data[idx]).float()
        
        return gene_data

# ============================================================================
# Load gene expression dataset for Gaussian VQ-VAE
# ============================================================================
def load_gene_expression_dataset(args):

    # Read the data, which contains smiles, inchikey, and gene values
    gene_data = pd.read_csv(args.gene_expression_file + args.pert_type +'.txt', 
                            sep='\t')

    # Mean target signatures across cell lines.
    if args.cell_name == "All":
        gene_data = gene_data.drop('cell_mfc_name',axis = 1
                            ).groupby(by = 'cmap_name'
                            ).mean().reset_index() # Average values for each protein
    elif args.cell_name == "AllCell":
        gene_data = gene_data.drop('cell_mfc_name', axis = 1) # Use all cell lines' signatures
    else:
        gene_data = gene_data[ gene_data['cell_mfc_name'] == args.cell_name 
                            ].drop('cell_mfc_name', axis = 1) # Select cell line.
        
    gene_data = gene_data.set_index( 'cmap_name' ) # Set index.

    # Normalize data per gene
    if args.gene_scaling == 'Orig':
        gene_data = gene_data # Original data
    elif args.gene_scaling == 'Std': 
        gene_data = (gene_data - gene_data.mean())/gene_data.std() # Diseaseはスパースなデータなので標準化
        gene_data = gene_data.dropna(how='any', axis = 1) # Remove genes with 0 for all diseases
    elif args.gene_scaling == 'Cent': 
        gene_data = (gene_data - gene_data.mean())
        gene_data = gene_data.dropna(how='any', axis = 1) # Remove genes with 0 for all diseases
    elif args.gene_scaling == 'MaxAbs':
        transformer = MaxAbsScaler() # Define transformer.
        X_scaled = transformer.fit_transform(gene_data) # MaxAbsScaler
        X_scaled = pd.DataFrame(X_scaled, index=gene_data.index, columns=gene_data.columns) # Numpy -> Pandas
        gene_data = X_scaled.replace(0, np.nan
                                ).dropna(how = 'all', axis = 1).fillna(0) # Remove genes with 0 values for all samples
        del X_scaled

    gene_data = gene_data.values.astype('float32') # Pandas -> numpy

    # Get a batch of gene data
    train_data = GeneExpressionDataset(gene_data)
    del gene_data
    gc.collect()

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.gene_batch_size, 
        shuffle=True
    )

    return train_loader

# ============================================================================
# Select cell lines based on missing rates.
# ============================================================================
def select_clls_based_on_missing_rates(args):

    # Missing rate data.
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

    return cell_lines


# ============================================================================
# Load gene expression dataset for Gaussian VQ-VAE integrating all cell lines
# ============================================================================
def load_gene_expression_dataset_MultiCell(args):

    # ===== Load gene expression data ===== #
    gene_data = pd.read_csv(args.gene_expression_file + args.pert_type + '.txt', sep='\t')
    all_gene_list = sorted(set(gene_data['cmap_name'])) # All gene list


    # ===== Select cell lines based on missing rates ====== #
    cell_lines = select_clls_based_on_missing_rates(args)
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

    # Get a batch of gene data
    train_data = GeneExpressionDataset(processed_gene_array)
    del gene_data, all_gene_list, processed_gene_data, cell_gene_data, cell_inputs
    gc.collect()

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.gene_batch_size, 
        shuffle=True
    )

    return train_loader, len(cell_lines)

# ============================================================================
# Load testing gene expression dataset
# ============================================================================
def load_test_gene_data(args):

    # Load data, which contains gene values
    data = pd.read_csv(
        args.test_gene_data + args.protein_name + '.csv', 
        sep=',',
        names=['name'] + ['gene'+str(i) for i in range(1,args.gene_num+1)]
    )
    
    data = data.iloc[:,1:]
    # Common the gene data with the columns of the source gene expression profiles
    data = common(data, args.gene_type)
    # Get a batch of gene data
    test_data = GeneExpressionDataset(data)
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.gene_batch_size, 
        shuffle=False
    )

    return test_loader

# ============================================================================
# Show density plots
# ============================================================================
def show_density(args, save_path,  nums, trained_gene_vae=None, idx=None):
    """
    save_path: the path to save the figure
    nums: number of rows of gene expression profile data used for data distribution
    """
    real_genes = pd.read_csv(args.gene_expression_file + args.pert_type + '.txt', sep='\t')
    
    # Mean target signatures across cell lines.
    if args.cell_name == "All":
        real_genes = real_genes.drop('cell_mfc_name', axis = 1
                                ).groupby(by = 'cmap_name'
                                ).mean().reset_index() # Average values for each protein
    elif args.cell_name == "AllCell":
        real_genes = real_genes.drop('cell_mfc_name', axis = 1) # Use all cell lines' signatures
    else:
        real_genes = real_genes[ real_genes['cell_mfc_name'] == args.cell_name 
                                ].drop('cell_mfc_name', axis = 1) # Select cell line.
    real_genes = real_genes.set_index( 'cmap_name' ) # Set index.
    
    # Normalize data per gene
    if args.gene_scaling == 'Orig':
        real_genes = real_genes # Original data
    elif args.gene_scaling == 'Std': 
        real_genes = (real_genes - real_genes.mean())/real_genes.std()
    elif args.gene_scaling == 'Cent': 
        real_genes = (real_genes - real_genes.mean())
    elif args.gene_scaling == 'MaxAbs':
        transformer = MaxAbsScaler() # Define transformer.
        X_scaled = transformer.fit_transform(real_genes) # MaxAbsScaler
        X_scaled = pd.DataFrame(X_scaled, index=real_genes.index, columns=real_genes.columns) # Numpy -> Pandas
        real_genes = X_scaled.replace(0, np.nan
                                ).dropna(how = 'all', axis = 1).fillna(0) # Remove genes with 0 values for all samples
        del X_scaled

    # Drop the nan row
    real_genes = real_genes.dropna(how='any')

    # Calculate average value
    if nums == 1:
        random_rows = np.array([idx])
    else:
        random_rows = np.random.choice(len(real_genes), nums)
        
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
        inputs = torch.tensor(real_genes.values, dtype=torch.float32)
        _, rec_genes, _ = trained_gene_vae(inputs) # [x,978]
        rec_genes = pd.DataFrame(rec_genes.cpu().detach().numpy())
        # Calculate average value
        mean_rec_gene = rec_genes.mean()
        # Figure density distribution
        sns.histplot(mean_rec_gene, bins=50, kde=True, label='Reconstructed gene', color='r')
    
    plt.legend()
    plt.savefig(save_path, dpi=150)

def show_all_gene_densities(args, trained_gene_vae):

    show_density(args, f"{make_output_directory_path(args)}/{args.all_gene_density_figure}", 10000, trained_gene_vae)

    for idx in range(10):
        show_density(args, f"{make_output_directory_path(args)}/one_gene_density_figure_{str(idx)}.png", 1, trained_gene_vae, idx)

# ============================================================================
# Load gene expression data for showing density plots
# ============================================================================
def show_density_load_gene_expression_dataset(args):

    """
    全細胞を並列に学習させた場合の結果を可視化するために、
    gene expression dataを読み込む
    """

    # ===== Load gene expression data ===== #
    gene_data = pd.read_csv(args.gene_expression_file + args.pert_type + '.txt', sep='\t')
    all_gene_list = sorted(set(gene_data['cmap_name'])) # All gene list


    # ===== Select cell lines based on missing rates ====== #
    cell_lines = select_clls_based_on_missing_rates(args)
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

    return processed_gene_array, cell_lines


def show_all_gene_densities_MultiCell(args, trained_gene_vae):
    """
    全細胞の情報を並列に学習させた場合

    <Input>
    save_path: the path to save the figure
    nums: number of rows of gene expression profile data used for data distribution
    """
    
    # ====== Load gene expression dataset ====== #
    processed_gene_array, cell_lines = show_density_load_gene_expression_dataset(args)

    nums_list = [1000] + [1]*10 # 複数タンパク質の平均, 各細胞のsignatureをplot
    idx = 0

    for nums in nums_list:

        # ===== Select proteins from real signatures ===== #
        if nums == 1:
            # random_rows = np.array([1])
            random_rows = np.array([idx])
        else:
            random_rows = np.random.choice(len(processed_gene_array), nums)
        real_genes = processed_gene_array[random_rows, :] # Select proteins.

        # ===== Reconstruct signatures ====== #
        if trained_gene_vae:
            trained_gene_vae.eval() # Evaluate mode
            inputs = torch.tensor(real_genes, dtype=torch.float32) # numpy array => tensor
            _, rec_genes, _ = trained_gene_vae(inputs) # [x,978]

        for i, cell in enumerate(cell_lines):

            # ===== Plot densities of original signatures. ===== #
            cell_real_genes = real_genes[:,i,:] # Select cell line.
            mean_real_all_gene = cell_real_genes.mean(axis = 0) # Calculate average value

            plt.subplots(figsize=(12,7))
            plt.title("Data distribution of gene expression profile", fontsize=12)
            plt.xlabel("Values of gene expression profile data", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            bins=np.histogram(mean_real_all_gene, bins=50)[1] #get the bin edges
            sns.histplot(mean_real_all_gene, bins=bins, kde=True, label='Real gene', color='g')
            # sns.histplot(mean_real_all_gene, bins=50, kde=True, label='Real gene', color='g')

            # ===== Plot densities of Reconstructed signatures ===== #
            cell_rec_genes = rec_genes[:,i,:] # Select cell line.
            cell_rec_genes = cell_rec_genes.cpu().detach().numpy()
            mean_rec_gene = cell_rec_genes.mean(axis = 0) # Calculate average value

            sns.histplot(mean_rec_gene, bins=bins, kde=True, label='Reconstructed gene', color='r')
            # sns.histplot(mean_rec_gene, bins=50, kde=True, label='Reconstructed gene', color='r')
            plt.legend()
            os.makedirs(f"{make_output_directory_path(args)}/plot", exist_ok = True)
            if nums == 1:
                save_path_cell = f"{make_output_directory_path(args)}/plot/{cell}_one_gene_density_figure_{str(idx-1)}.png"
            else:
                save_path_cell = f"{make_output_directory_path(args)}/plot/{cell}_all_gene_density_figure.png"
            plt.savefig(save_path_cell, dpi=150)

        idx += 1

# def show_all_gene_densities_MultiCell(args, trained_gene_vae):

#     show_density_MultiCell(args, f"{make_output_directory_path(args)}/", 10000, trained_gene_vae)

#     for idx in range(10):
#         show_density_MultiCell(args, f"{make_output_directory_path(args)}/one_gene_density_figure_{str(idx)}.png", 1, trained_gene_vae, idx)

# ============================================================================










