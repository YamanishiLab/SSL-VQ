import torch
import numpy as np
import pandas as pd
import gc # garbage collection
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MaxAbsScaler

from GeneVAE_disease import GeneVAE
from utils_disease import get_device, common, make_output_directory_path

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
        args.gene_expression_file + args.profile_type, 
        sep='\t',
        index_col='disease'
    )

    # Normalize data per gene
    if args.gene_scaling == 'Orig':
        data = data # Original data
    elif args.gene_scaling == 'Std': 
        data = (data - data.mean())/data.std() # Diseaseはスパースなデータなので標準化
        data = data.dropna(how='any', axis = 1) # Remove genes with 0 for all diseases
    elif args.gene_scaling == 'MaxAbs':
        transformer = MaxAbsScaler() # Define transformer.
        X_scaled = transformer.fit_transform(data) # MaxAbsScaler
        X_scaled = pd.DataFrame(X_scaled, index=data.index, columns=data.columns) # Numpy -> Pandas
        data = X_scaled.replace(0, np.nan
                                ).dropna(how = 'all', axis = 1).fillna(0) # Remove genes with 0 values for all samples
        del X_scaled

    data = data.values.astype('float32') # Pandas -> numpy
    gene_num = data.shape[1] # Number of genes

    # Get a batch of gene data
    train_data = GeneExpressionDataset(data)
    del data
    gc.collect()

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.gene_batch_size, 
        shuffle=True
    )

    return train_loader, gene_num

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
    train_loader, gene_num = load_gene_expression_dataset(args)

    # Activation function
    if args.gene_activation_fn == 'ReLU':
        gene_activation_fn = nn.ReLU()
    elif args.gene_activation_fn == 'Tanh':
        gene_activation_fn = nn.Tanh()
    
    # Define GeneVAE 
    gene_vae = GeneVAE(
        input_size=gene_num, 
        hidden_sizes=args.gene_hidden_sizes,
        latent_size=args.gene_latent_size,
        output_size=gene_num,
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
        mean_rec_loss = total_rec_loss / (len(train_loader.dataset) * gene_num)
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



























