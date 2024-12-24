import torch
import numpy as np
import pandas as pd
import gc # garbage collection
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from GeneVAE_target import GeneVAE
from utils_target import get_device, common, make_output_directory_path

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
# Load gene expression dataset
def load_gene_expression_dataset(args):
    
    # Load data, which contains smiles, inchikey, and gene values
    data = pd.read_csv(
        args.gene_expression_file + args.pert_type +'.txt', 
        sep='\t' 
    )

    # Mean target signatures across cell lines
    if args.cell_name == "All":
        data = data.drop('cell_mfc_name', axis = 1
                    ).groupby(by = 'cmap_name'
                    ).mean().reset_index() # Average gene expression signatures across all cell lines.
    elif args.cell_name == "All_AllCell":
        averaged_data = data.drop('cell_mfc_name', axis = 1
                                  ).groupby(by = 'cmap_name').mean().reset_index() # Average values for each protein
        averaged_data.insert( 1, 'cell_mfc_name', 'All' ) # add cell line columns

        data = pd.concat([averaged_data, data], axis=0) # Concatenate averaged data and original data.
        data['cmap_name'] = [ f"{a}@{b}" 
                             for a,b in zip(data['cmap_name'], data['cell_mfc_name']) ] # Merge protein names and cell lines
        data = data.drop('cell_mfc_name', axis = 1)
        del averaged_data
        gc.collect()
    else:
        data = data[ data['cell_mfc_name'] == args.cell_name
                    ].drop('cell_mfc_name', axis = 1) # Select cell line.
    data = data.set_index( 'cmap_name' ) # indexを指定
    data = data.values.astype('float32') # Pandas -> numpy

    # # Drop the nan row
    # data = data.dropna(how='any')

    # Normalize data per gene 
    #data = (data - data.mean())/data.std()

    # Get a batch of gene data
    train_data = GeneExpressionDataset(data)
    del data
    gc.collect()

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.gene_batch_size, 
        shuffle=True
        # shuffle = False
    )

    return train_loader

# ============================================================================
# Load testing gene expression dataset
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
# Train GeneVAE
def train_gene_vae(args):
  
    # Load gene dataset
    train_loader = load_gene_expression_dataset(args)

    # Activation function
    if args.gene_activation_fn == 'ReLU':
        gene_activation_fn = nn.ReLU()
    elif args.gene_activation_fn == 'Tanh':
        gene_activation_fn = nn.Tanh()
    
    # Define GeneVAE 
    gene_vae = GeneVAE(
        input_size=args.gene_num, 
        hidden_sizes=args.gene_hidden_sizes,
        latent_size=args.gene_latent_size,
        output_size=args.gene_num,
        activation_fn=gene_activation_fn,
        # activation_fn=nn.Tanh(), # Revise: ReLU -> Tanh
        # activation_fn=nn.ReLU(),
        dropout=args.gene_dropout
    ).to(get_device())

    # Optimizer
    gene_optimizer = optim.Adam(gene_vae.parameters(), lr=args.gene_lr)

    # Gradually decrease the alpha (weight of MSE relative to KL)
    alpha = 0.5
    alphas = torch.cat([
        torch.linspace(0.99, alpha, int(args.gene_epochs/2)), 
        alpha * torch.ones(args.gene_epochs - int(args.gene_epochs/2))
    ]).double().to(get_device())

    # Prepare file to save results
    # o_dir = f"../data/VAE/{args.pert_type}/CellLine_{args.cell_name}/results/"
    # os.makedirs(o_dir, exist_ok = True )
    with open( f"{ make_output_directory_path(args) }/{args.gene_vae_train_results}", 'a+') as wf:
        wf.truncate(0)
        wf.write('{},{},{},{}\n'.format('Epoch', 'Joint', 'Rec', 'KLD'))

    print('Training Information:')
    for epoch in range(args.gene_epochs):

        total_joint_loss = 0
        total_rec_loss = 0
        total_kld_loss = 0
        gene_vae.train()
        
        for _, genes in enumerate(train_loader):

            genes = genes.to(get_device())
            _, rec_genes = gene_vae(genes)
            joint_loss, rec_loss, kld_loss = gene_vae.joint_loss(
                outputs=rec_genes, 
                targets=genes,
                alpha=alphas[epoch],
                beta=1.
            )

            gene_optimizer.zero_grad()
            joint_loss.backward()
            gene_optimizer.step()

            total_joint_loss += joint_loss.item()
            total_rec_loss += rec_loss.item()
            total_kld_loss += kld_loss.item()
        
        mean_joint_loss = total_joint_loss / len(train_loader.dataset)
        mean_rec_loss = total_rec_loss / (len(train_loader.dataset) * args.gene_num)
        mean_kld_loss = total_kld_loss / (len(train_loader.dataset) * args.gene_latent_size)
        print('Epoch {:d} / {:d}, joint_loss: {:.3f}, rec_loss: {:.3f}, kld_loss: {:.3f},'.format(\
            epoch+1, args.gene_epochs, mean_joint_loss, mean_rec_loss, mean_kld_loss))
        
        # Save trained results to file
        with open( f"{make_output_directory_path(args)}/{args.gene_vae_train_results}", 'a+') as wf:
            wf.write('{},{:.3f},{:.3f},{:.3f}\n'.format(epoch+1, mean_joint_loss, mean_rec_loss, mean_kld_loss))
        
    # Save trained GeneVAE
    gene_vae.save_model( make_output_directory_path(args) + args.saved_gene_vae + '.pkl')
    print('Trained GeneVAE is saved in {}'.format( make_output_directory_path(args) + args.saved_gene_vae + '.pkl' ))

    return gene_vae



























