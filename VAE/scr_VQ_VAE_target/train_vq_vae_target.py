import torch
import numpy as np
import pandas as pd
import gc # gabage collection
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MaxAbsScaler

from VQ_VAE_target import VQ_VAE, VQ_VAE_MultiCells
from utils_vq_vae_target import get_device, make_output_directory_path, load_gene_expression_dataset, \
load_gene_expression_dataset_MultiCell

# ============================================================================
# Activation function.
# ============================================================================
def get_activation_fn(args):
    # Activation function
    if args.gene_activation_fn == 'ReLU':
        gene_activation_fn = nn.ReLU()
    elif args.gene_activation_fn == 'Tanh':
        gene_activation_fn = nn.Tanh()
    elif args.gene_activation_fn == 'SELU':
        gene_activation_fn = nn.SELU()
    elif 'LeakyReLU' in args.gene_activation_fn:
        negative_slope = float(args.gene_activation_fn.replace('LeakyReLU',''))
        gene_activation_fn = nn.LeakyReLU(negative_slope=negative_slope)
    return gene_activation_fn

# ============================================================================
# Train VQ-VAE
# ============================================================================
def train_vq_vae(args):

    ## Activation function
    gene_activation_fn = get_activation_fn(args)
  
    # Load gene dataset
    if args.cell_name != 'AllCellParallel':
        train_loader = load_gene_expression_dataset(args)
        vq_vae = VQ_VAE(
            gene_num=args.gene_num,
            num_emb=args.gene_num_emb,  
            emb_dim=args.gene_emb_dim,   
            com_cost=args.gene_com_cost,
            activation_fn=gene_activation_fn,
            dropout=args.gene_dropout
        ).to(get_device())

    elif args.cell_name == 'AllCellParallel':
        train_loader, num_cell_lines = load_gene_expression_dataset_MultiCell(args)
        vq_vae = VQ_VAE_MultiCells(
            gene_num=args.gene_num,
            num_emb=args.gene_num_emb,  
            emb_dim=args.gene_emb_dim,   
            com_cost=args.gene_com_cost,
            activation_fn=gene_activation_fn,
            dropout=args.gene_dropout,
            num_cell_lines=num_cell_lines
        ).to(get_device())
    
    # Optimizer
    gene_optimizer = optim.Adam(vq_vae.parameters(), lr=args.gene_lr, amsgrad=True)

    train_res_recon_error = []
    train_loss = []
    train_vq_loss = []

    # Prepare file to save results
    with open( f"{ make_output_directory_path(args) }/{args.gene_vae_train_results}", 'a+') as wf:
        wf.truncate(0)
        wf.write('{},{},{},{}\n'.format('Epoch', 'Total', 'Rec', 'Vq'))

    print('Training Information:')
    for epoch in range(args.gene_epochs):

        count =0
        vq_vae.train()

        for _, inputs in enumerate(train_loader):
            count +=1
            gene_optimizer.zero_grad()
            inputs = inputs.to(get_device())
            vq_loss, data_recon, _ = vq_vae(inputs)
            recon_error = F.mse_loss(data_recon, inputs)
            loss = recon_error + vq_loss
            loss.backward()
            gene_optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_loss.append(loss.item())
            train_vq_loss.append(vq_loss.item())

        ave_error = np.mean(train_res_recon_error)
        ave_vq = np.mean(train_vq_loss)
        ave_loss = np.mean(train_loss)
        print('Epoch {:d},  recon_error: {:.3f}, vq_loss: {:.3f}, total_loss: {:.3f}'.format(epoch+1, ave_error, ave_vq, ave_loss))
        
        # Save trained results to file
        with open( f"{make_output_directory_path(args)}/{args.gene_vae_train_results}", 'a+') as wf:
            wf.write('{},{:.3f},{:.3f},{:.3f}\n'.format(epoch+1, ave_loss, ave_error, ave_vq))
        
        # show_one_gene_densities(model)

    # Save trained VQ-VAE
    vq_vae.save_model( make_output_directory_path(args) + args.saved_gene_vae + '.pkl')
    print('Trained VQ-VAE is saved in {}'.format( make_output_directory_path(args) + args.saved_gene_vae + '.pkl' ))

    return vq_vae



























