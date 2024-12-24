import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# import sys
# sys.path.insert(2, 'codes_target')
from utils_vq_vae_target import get_device

# ============================================================================
# Quantize tensors using VQ-VAE (Nearest K Neighbors)
class VectorQuantizer(nn.Module):
    
    def __init__(self, num_emb, emb_dim, com_cost):
        """
        num_emb (K): the number of vectors in the quantized space.
        emb_dim (d): the dimensionality of the tensors in the quantized space.
        com_cost: scalar which controls the weighting of the loss terms
        """
        super(VectorQuantizer, self).__init__()
        
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.com_cost = com_cost
        self.emb = nn.Embedding(self.num_emb, self.emb_dim)
        self.emb.weight.data.uniform_(-1/self.num_emb, 1/self.num_emb)
        
    def forward(self, inputs):
        
        # Distance
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.emb.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.emb.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_emb, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.emb.weight).view(inputs.shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.com_cost * e_latent_loss

         # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        
        return loss, quantized, encodings
    
# ============================================================================
# Create a VQ-VAE model
# ============================================================================
class VQ_VAE(nn.Module):
    
    def __init__(
        self,  
        gene_num,
        num_emb, 
        emb_dim, 
        com_cost,
        activation_fn,
        dropout
    ):
        super(VQ_VAE, self).__init__()
        
        """
        encoder_layers = []
        for index in range(1, len(emb_dim)):
            if index == 1:
                encoder_layers.append(nn.Linear(GENE_NUM, emb_dim[index-1]))
            else:
               encoder_layers.append(nn.Linear(emb_dim[index-1], emb_dim[index]))
            encoder_layers.append(nn.Tanh())
            encoder_layers.append(nn.Dropout(0.1))
        self.encoder = nn.Sequential(*encoder_layers)
        """

        self.activation_fn = activation_fn
        self.dropout_fn = nn.Dropout(p=dropout)
        
        self.encoder =  nn.Sequential(
            nn.Linear(gene_num, emb_dim[0]),
            self.activation_fn,
            self.dropout_fn,
            nn.Linear(emb_dim[0], emb_dim[1]),
            self.activation_fn,
            self.dropout_fn,
            nn.Linear(emb_dim[1], emb_dim[2]),
            self.activation_fn,
            self.dropout_fn
        )
       
        self.vq = VectorQuantizer(num_emb, emb_dim[-1], com_cost)

        """
        decoder_layers = []
        for index in range(len(emb_dim)-1, -1, -1):
            if index != 0:
                decoder_layers.append(nn.Linear(emb_dim[index], emb_dim[index-1]))
                decoder_layers.append(nn.Tanh())
                decoder_layers.append(nn.Dropout(0.1))
            else:
                decoder_layers.append(nn.Linear(emb_dim[index], GENE_NUM))
        self.decoder = nn.Sequential(*decoder_layers)
        """
        self.decoder =  nn.Sequential(
            nn.Linear(emb_dim[-1], emb_dim[-2]),
            self.activation_fn,
            self.dropout_fn,
            nn.Linear(emb_dim[-2], emb_dim[-3]),
            self.activation_fn,
            self.dropout_fn,
            nn.Linear(emb_dim[-3], gene_num)
        )
            
    def forward(self, inputs):
        z = self.encoder(inputs)
        loss, quantized, _ = self.vq(z)
        x_recon = self.decoder(quantized) 

        return loss, x_recon, quantized
    
    def load_model(self, path):
        weights = torch.load(path, map_location=get_device())
        self.load_state_dict(weights) # Revise: delete "strict=False" (Model Changed!)

    def save_model(self, path):
        torch.save(self.state_dict(), path) # Revise: checkpoint -> state_dict()


# ============================================================================
# Create a VQ-VAE model with all cell lines.
# ============================================================================

class VQ_VAE_MultiCells(nn.Module):
    
    def __init__(
        self,  
        gene_num,
        num_emb, 
        emb_dim, 
        com_cost,
        activation_fn,
        dropout,
        num_cell_lines
    ):
        super(VQ_VAE_MultiCells, self).__init__()
        
        """
        encoder_layers = []
        for index in range(1, len(emb_dim)):
            if index == 1:
                encoder_layers.append(nn.Linear(GENE_NUM, emb_dim[index-1]))
            else:
               encoder_layers.append(nn.Linear(emb_dim[index-1], emb_dim[index]))
            encoder_layers.append(nn.Tanh())
            encoder_layers.append(nn.Dropout(0.1))
        self.encoder = nn.Sequential(*encoder_layers)
        """

        self.activation_fn = activation_fn
        self.dropout_fn = nn.Dropout(p=dropout)
        self.emb_dim = emb_dim

        if len(emb_dim) == 3:

            # ===== Encoder ===== #
            # Each cell line layer.
            self.each_cell_encoder = nn.ModuleList()
            for _ in range(num_cell_lines):
                cell_specific_encoder = nn.Sequential(
                    nn.Linear(gene_num, emb_dim[0]),
                    self.activation_fn,
                    self.dropout_fn
                )
                self.each_cell_encoder.append(cell_specific_encoder)
            # for _ in range(num_cell_lines):
            #     self.each_cell_encoder.append(nn.Linear(gene_num, emb_dim[0]))

            # Cell line integrated layers.
            self.integrated_encoder =  nn.Sequential(
                nn.Linear(num_cell_lines*emb_dim[0], emb_dim[1]),
                self.activation_fn,
                self.dropout_fn,
                nn.Linear(emb_dim[1], emb_dim[2]),
                self.activation_fn,
                self.dropout_fn
            )

            # ===== Decoder ===== #
            # Cell line integrated decoders.
            self.integrated_decoder =  nn.Sequential(
                nn.Linear(emb_dim[-1], emb_dim[-2]),
                activation_fn,
                self.dropout_fn,
                nn.Linear(emb_dim[-2], num_cell_lines*emb_dim[-3]),
                activation_fn,
                self.dropout_fn
            )        
                
            # Each cell line layer.
            self.each_cell_decoder = nn.ModuleList()
            for _ in range(num_cell_lines):
                self.each_cell_decoder.append(nn.Linear(emb_dim[-3], gene_num))

        elif len(emb_dim) == 4:

            # ===== Encoder ===== #
            # Each cell line layer.
            self.each_cell_encoder = nn.ModuleList()
            for _ in range(num_cell_lines):
                cell_specific_encoder = nn.Sequential(
                    nn.Linear(gene_num, emb_dim[0]),
                    self.activation_fn,
                    self.dropout_fn,
                    nn.Linear(emb_dim[0], emb_dim[1]),
                    self.activation_fn,
                    self.dropout_fn
                )
                self.each_cell_encoder.append(cell_specific_encoder)

            # Cell line integrated layers.
            self.integrated_encoder =  nn.Sequential(
                nn.Linear(num_cell_lines*emb_dim[1], emb_dim[2]),
                self.activation_fn,
                self.dropout_fn,
                nn.Linear(emb_dim[2], emb_dim[3]),
                activation_fn,
                self.dropout_fn
            )

            # ===== Decoder ===== #
            # Cell line integrated decoders.
            self.integrated_decoder =  nn.Sequential(
                nn.Linear(emb_dim[-1], emb_dim[-2]),
                activation_fn,
                self.dropout_fn,
                nn.Linear(emb_dim[-2], num_cell_lines*emb_dim[-3]),
                activation_fn,
                self.dropout_fn
            )        
                
            # Each cell line layer.
            self.each_cell_decoder = nn.ModuleList()
            for _ in range(num_cell_lines):
                cell_specific_decoder = nn.Sequential(
                    nn.Linear(emb_dim[-3], emb_dim[-4]),
                    self.activation_fn,
                    self.dropout_fn,
                    nn.Linear(emb_dim[-4], gene_num)
                )
                self.each_cell_decoder.append(cell_specific_decoder)
       
        # ===== VQ quantizer ===== #
        self.vq = VectorQuantizer(num_emb, emb_dim[-1], com_cost)

        """
        decoder_layers = []
        for index in range(len(emb_dim)-1, -1, -1):
            if index != 0:
                decoder_layers.append(nn.Linear(emb_dim[index], emb_dim[index-1]))
                decoder_layers.append(nn.Tanh())
                decoder_layers.append(nn.Dropout(0.1))
            else:
                decoder_layers.append(nn.Linear(emb_dim[index], GENE_NUM))
        self.decoder = nn.Sequential(*decoder_layers)
        """


    def forward(self, inputs):
        
        # Each cell line encoders.
        each_cell_out = []
        for idx, enc in enumerate(self.each_cell_encoder):
            each_cell_out.append(enc(inputs[:,idx]))
        # for idx, enc in enumerate(self.each_cell_encoder):
        #     out = enc(inputs[:,idx])
        #     out = self.activation_fn(out)
        #     out = self.dropout_fn(out)
        #     each_cell_out.append(out)
        each_cell_out = torch.cat(each_cell_out, dim=1)

        # Integrated encoders.
        z = self.integrated_encoder(each_cell_out)

        # Quantizer.
        loss, quantized, _ = self.vq(z)

        # Integrated decoders.
        integrated_x_recon = self.integrated_decoder(quantized) 
        integrated_x_recon = torch.split(integrated_x_recon, self.emb_dim[-3], dim=1)

        # Each cell line decoders.
        x_recon = []
        for idx, dec in enumerate(self.each_cell_decoder):
            x_recon.append(dec(integrated_x_recon[idx]))
        x_recon = torch.stack( x_recon, dim=1) # list => torch.tensor

        return loss, x_recon, quantized
    
    def load_model(self, path):
        weights = torch.load(path, map_location=get_device())
        self.load_state_dict(weights) # Revise: delete "strict=False" (Model Changed!)

    def save_model(self, path):
        torch.save(self.state_dict(), path) # Revise: checkpoint -> state_dict()

# ============================================================================







