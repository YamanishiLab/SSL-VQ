import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import argparse
import os
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryAUPRC

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
# Consistency loss 
def consistency_loss(logits_w1, logits_w2):
    logits_w2 = logits_w2.detach()
    assert logits_w1.size() == logits_w2.size()
    # return F.mse_loss(torch.softmax(logits_w1, dim=-1), torch.softmax(logits_w2, dim=-1), reduction='mean')
    return F.mse_loss( torch.sigmoid(logits_w1)[:,0], torch.sigmoid(logits_w2)[:,0], reduction='mean')

# ============================================================================
# Focal multiclass loss 
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
      super(FocalLoss, self).__init__()
      self.gamma = gamma
      self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
      self.reduction = reduction
      self.bceloss = nn.BCELoss(reduction=self.reduction, weight=self.weight)

    def forward(self, outputs, targets): 
      bceloss = self.bceloss(outputs, targets)
      pt = torch.exp(-bceloss)
      focal_loss = ((1 - pt) ** self.gamma * bceloss).mean()
      return focal_loss

# ============================================================================
# Performance evaluation based on accuracy, AUC and AUPR 
def performance_eval(probas, label):
    """
    outputs: outputs of the model [batch_size, output_size]
    label: class labels for protein-disease pairs in the batch data [batch_size, 1]
    """

    # Accuracy
    acc_metric = BinaryAccuracy()
    acc_metric.update(probas, label)
    acc=acc_metric.compute()

    # AUC
    auc_metric = BinaryAUROC()
    auc_metric.update(probas, label)
    auc=auc_metric.compute()

    # AUPR
    aupr_metric = BinaryAUPRC()
    aupr_metric.update(probas, label)
    aupr=aupr_metric.compute()

    return acc, auc, aupr

# ============================================================================
# Early stopping from github code: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ============================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
def get_input_directory_path(args):

    # Meke output data directory
    if args.disease_vae_type == 'Original' and args.gene_vae_type == 'Original':
        i_dir = f"../data/{args.pert_type}/DiseaseFeaeture_{args.disease_feature_type}/" # Disease feature type
        i_dir = f"{i_dir}/TargetFeaeture_{args.target_feature_type}/" # Target feature type
    else:
        i_dir = f"../data/{args.pert_type}/DiseaseFeaeture_{args.disease_vae_type}_{args.disease_feature_type}/" # Disease feature type
        i_dir = f"{i_dir}/TargetFeaeture_{args.target_vae_type}_{args.target_feature_type}/" # Target feature type
    i_dir = f"{i_dir}/CellLine_{args.cell_name}/" # Cell line
    i_dir = f"{i_dir}/PairD{args.disease_select}T{args.target_select}/" # Unlabeled data
    return i_dir

# ============================================================================
def make_output_directory_path(args):

    # Meke output data directory
    if args.disease_vae_type == 'Original' and args.gene_vae_type == 'Original':
        o_dir = f"../data/{args.pert_type}/DiseaseFeaeture_{args.disease_feature_type}/" # Disease feature type
        o_dir = f"{o_dir}/TargetFeaeture_{args.target_feature_type}/" # Target feature type
    else:
        o_dir = f"../data/{args.pert_type}/DiseaseFeaeture_{args.disease_vae_type}_{args.disease_feature_type}/" # Disease feature type
        o_dir = f"{o_dir}/TargetFeaeture_{args.target_vae_type}_{args.target_feature_type}/" # Target feature type

    o_dir = f"{o_dir}/CellLine_{args.cell_name}/" # Cell line

    o_dir = f"{o_dir}/PairD{args.disease_select}T{args.target_select}/" # Unlabeled data

    o_dir = f"{o_dir}/Epo{args.gene_epochs}_Lr{args.gene_lr}" # Epochs
    o_dir = f"{o_dir}_Hid{'_'.join([str(s) for s in sorted(args.gene_hidden_sizes, reverse = True)])}" # hidden sizes
    o_dir = f"{o_dir}_Bat{args.gene_batch_size}" # batch size
    o_dir = f"{o_dir}_Dro{args.gene_dropout}" # drop out
    o_dir = f"{o_dir}_Los{args.loss_fn}" # loss function
    o_dir = f"{o_dir}_EStop{args.early_stopping}/" # early stopping
    o_dir = f"{o_dir}/foldf{args.fold_number}/" # fold number
    os.makedirs( o_dir, exist_ok = True ) # make directory
    return o_dir

# ============================================================================
def show_gene_vae_hyperparamaters(args):

    # Hyper-parameters
    params = {}
    print('\n\nConcatNN Hyperparameter Information:')
    print('='*50)
    params['PERTURBATION_TYPE'] = args.pert_type 
    params['CELL_NAME'] = args.cell_name
    params['FOLD_NUMBER'] = args.fold_number
    params['TARGET_FEATURE_TYPE'] = args.target_feature_type
    params['DISEASE_PROFILE_TYPE'] = args.disease_profile_type
    params['DISEASE_FEATURE_TYPE'] = args.disease_feature_type
    params['GENE_EPOCHS'] = args.gene_epochs 
    params['GENE_LR'] = args.gene_lr
    # params['GENE_NUM'] = args.gene_num
    params['GENE_HIDDEN_SIZES'] = args.gene_hidden_sizes
    params['GENE_OUTPUT_SIZE'] = args.gene_output_size
    params['GENE_BATCH_SIZE'] = args.gene_batch_size
    params['GENE_DROUPOUT'] = args.gene_dropout
    params['LOSS_FUNCTION'] = args.loss_fn
    params['EARLLY_STOPPING'] = args.early_stopping
    params['TARGET_VAE_TYPE'] = args.target_vae_type
    params['DISEASE_VAE_TYPE'] = args.disease_vae_type
    params['DISEASE_UNLABELED_DATA'] = args.disease_select
    params['TARGET_UNLABELED_DATA'] = args.target_select

    for param in params:
        string = param + ' ' * (5 - len(param))
        print('{}:   {}'.format(string, params[param]))
    print('='*50)

# ============================================================================
# Exponential moving average
class EMA:
    """
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ============================================================================
# Batch normalization controller
class Bn_Controller:
    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + '.running_mean'] = m.running_mean.data.clone()
                self.backup[name + '.running_var'] = m.running_var.data.clone()
                self.backup[name + '.num_batches_tracked'] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + '.running_mean']
                m.running_var.data = self.backup[name + '.running_var']
                m.num_batches_tracked.data = self.backup[name + '.num_batches_tracked']
        self.backup = {}

# ============================================================================










