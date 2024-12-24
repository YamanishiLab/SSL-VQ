import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# import sys
# sys.path.insert(2, '../scr_train/')
from utils_pimodel import get_device

# ============================================================================
# Create a NN model
class ConcatNN(nn.Module):
    def __init__(
        self, 
        input_size,
        hidden_sizes,
        output_size,
        activation_fn,
        dropout
    ):
        """
        input_size: number of gene columns (eg. 15,782)
        hidden_sizes: number of neurons of stack dense layers
        activation_fn: activation function
        dropout: dropout probabilites
        """
        super(ConcatNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_fn = activation_fn
        self.dropout = [dropout] * len(self.hidden_sizes)

        num_units = [self.input_size] + self.hidden_sizes
        
        dense_layers = []
        for index in range(1, len(num_units)):
            dense_layers.append(nn.Linear(num_units[index-1], num_units[index]))
            dense_layers.append(self.activation_fn)

            if self.dropout[index-1] > 0.0:
                dense_layers.append(nn.Dropout(p=self.dropout[index-1]))
        else:
            dense_layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))
            # dense_layers.append(nn.Sigmoid())

        self.linear_rerul_stack = nn.Sequential(*dense_layers)

        # self.encoding_to_mu = nn.Linear(self.hidden_sizes[-1], self.latent_size)
        # self.encoding_to_logvar = nn.Linear(self.hidden_sizes[-1], self.latent_size)

    def forward(self, inputs):
        """
        inputs: [batch_size, input_size]
        returns: 
            mu: [batch_size, latent_size]
            logvar: [batch_size, latent_size]
        """
        logits = self.linear_rerul_stack(inputs)
        # mu = self.encoding_to_mu(projection)
        # logvar = self.encoding_to_logvar(projection)

        return logits
    
    def load_model(self, path):
        weights = torch.load(path, map_location=get_device())
        self.load_state_dict(weights) # Revise: delete "strict=False" (Model Changed!)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        # torch.save( checkpoint, path)

















