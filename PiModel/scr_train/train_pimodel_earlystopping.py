from cgi import test
from random import shuffle
import numpy as np
import pandas as pd
import gc # gabage collection
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import *

from ConcatNN import ConcatNN
from utils_pimodel import get_device, get_input_directory_path, make_output_directory_path, performance_eval, \
    EarlyStopping, FocalLoss, consistency_loss, EMA, Bn_Controller

# ============================================================================
# Define gene expression dataset
class GeneExpressionDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, lb_or_ulb):

        self.data = data
        self.data_num = len(data)
        self.lb_or_ulb=lb_or_ulb

    def __len__(self):

        return self.data_num

    def __getitem__(self, idx):
        """
        idx: sample index.
        lb_or_ulb: 'lb' means labeled data, and 'ulb' means unlabeled data.
        """

        if self.lb_or_ulb=='lb':
            # Feaeture data
            gene_data = self.data.T[1:,:].T # labelの列を除く
            gene_data = torch.tensor(gene_data[idx]).float()

            # Label data
            pair_label_data = torch.tensor(self.data.T[0].T[idx],
                                        ).float()
            return gene_data, pair_label_data
        
        elif self.lb_or_ulb=='ulb':
            # Feaeture data
            gene_data1, gene_data2 = np.hsplit(self.data, 2) 
            gene_data1 = torch.tensor(gene_data1[idx]).float()  
            gene_data2 = torch.tensor(gene_data2[idx]).float()            
            return gene_data1, gene_data2

# ============================================================================
# Load gene expression dataset
def load_gene_expression_dataset(args):
    
    # Labeled feature expression data.
    l_data = pd.read_csv(get_input_directory_path(args) + "/feature_files/l_feature_expression_cv.tabs",
                        sep='\t',
                        index_col=0
                        ).astype('float32')
    
    # Unlabeled feature expression data.
    u_data1 = pd.read_csv(get_input_directory_path(args) + "/feature_files/u1_feature_expression_cv.tabs",
                        sep='\t',
                        index_col=0
                        ).astype('float32')
    
    # Unlabeled feature expression data.
    u_data2 = pd.read_csv(get_input_directory_path(args) + "/feature_files/u2_feature_expression_cv.tabs",
                        sep='\t',
                        index_col=0
                        ).astype('float32')
    
    # Label data.
    train_y = pd.read_csv( get_input_directory_path(args) + f"/training_files/training_label_foldf{args.fold_number}.tabs",
                        sep='\t',
                        index_col='pair_index'
                        ).astype('int32')
    
    # Input size.
    input_size = l_data.shape[1]
    
    # Class weight
    label_ratio = train_y.value_counts("label", sort=False, normalize=True) # class label ratio
    class_weight = torch.tensor(1/label_ratio).clone().to(get_device(), torch.float32) # class weight
    pos_weight = class_weight[1] / class_weight[0] # positive calss weight

    # Merge feature and label data.
    l_data = pd.merge( train_y, l_data, left_index=True, right_index=True, how = 'left' ).astype('float32')

    # Merge unlabeled data.
    u_data = pd.merge( u_data1, u_data2, left_index=True, right_index=True)

    # Drop the nan row
    l_data = l_data.dropna(how='any')
    u_data = u_data.dropna(how='any')

    # Pandas => Numpy array
    l_data = l_data.values.astype('float32')
    u_data = u_data.values.astype('float32')

    # Normalize data per gene 
    #data = (data - data.mean())/data.std()

    # Get a batch of gene data
    l_train_data = GeneExpressionDataset(l_data, lb_or_ulb='lb')
    u_train_data = GeneExpressionDataset(u_data, lb_or_ulb='ulb')
    del train_y, l_data, u_data1, u_data2, u_data
    gc.collect()

    l_train_loader = torch.utils.data.DataLoader(
        l_train_data, 
        batch_size=args.gene_batch_size, 
        shuffle=True,
        pin_memory=True,
    )
    u_train_loader = torch.utils.data.DataLoader(
        u_train_data, 
        batch_size=args.gene_batch_size, 
        shuffle=True,
        pin_memory=True,
    )

    return l_train_loader, u_train_loader, pos_weight, input_size

# ============================================================================
# Load testing gene expression dataset
def load_test_gene_data(args):

    # Labeled feature expression data.
    data = pd.read_csv(get_input_directory_path(args) + f"/test_files/test_feature_expression_foldf{args.fold_number}.tabs",
                        sep='\t',
                        index_col=0
                        ).astype('float32')
    
    # Label data.
    test_y = pd.read_csv(get_input_directory_path(args) + f"/test_files/test_label_foldf{args.fold_number}.tabs",
                        sep='\t',
                        index_col='pair_index'
                        ).astype('int32')

    # Merge feature and label data.
    data = pd.merge( test_y, data, left_index=True, right_index=True, how = 'left' ).astype('float32')

    # Drop the nan row
    data = data.dropna(how='any')

    # Pandas => Numpy array
    data = data.values.astype('float32')
    # test_batch_size = data.shape[0]

    # Get a batch of gene data
    test_data = GeneExpressionDataset(data, lb_or_ulb='lb')
    del data
    gc.collect()

    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.gene_batch_size, 
        # batch_size=test_batch_size, # test dataをバッチごとの分けていると時間がかかるので、分割しない
        shuffle = True,
        pin_memory=True,
    )

    # return data, test_y
    return test_loader

# ============================================================================
# Train PiModel
def train_pi_model_earlystopping(args):  
    """
    l_train_loaderとu_train_loaderの長さが違うため、
    長い方（通常はu_train_loader）のloaderの一部が使われないことに注意。
    """

    # Load gene dataset
    l_train_loader, u_train_loader, pos_weight, input_size = load_gene_expression_dataset(args)
    test_loader = load_test_gene_data(args)

    # Define ConcatNN
    gene_nn = ConcatNN(
        input_size=input_size, 
        hidden_sizes=args.gene_hidden_sizes,
        output_size=args.gene_output_size,
        activation_fn=nn.ReLU(),
        dropout=args.gene_dropout
    ).to(get_device())

    # EMA model
    gene_ema = EMA(gene_nn, args.ema_m)

    # Optimizer
    gene_optimizer = optim.Adam(gene_nn.parameters(), lr=args.gene_lr)

    # Batch normalization controller
    gene_bn_controller = Bn_Controller()

    # Loss function
    if args.loss_fn == "CEL":
        gene_loss_function = nn.BCELoss()
    elif args.loss_fn == "CELweight":
        gene_loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss_fn == "FL":
        gene_loss_function = FocalLoss()

    # Iteration
    it = 0

    # Prepare file to save results
    with open( f"{ make_output_directory_path(args) }/{args.gene_nn_train_results}", 'a+') as wf:
        wf.truncate(0)
        wf.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('Epoch', 
                'Train_Total', 'Train_Sup', 'Train_Unsup', 'Train_Accuracy', 'Train_AUC', 'Train_AUPR',
                'Test_Total', 'Test_Accuracy', 'Test_AUC', 'Test_AUPR'))

    print('Training Information:')
    early_stopping = EarlyStopping(patience=int(args.early_stopping),
                                    path=make_output_directory_path(args) + args.saved_gene_nn + '.pkl',
                                    ) # Early stopping.
    for epoch in range(args.gene_epochs):
        
        # lb: labeled, ulb: unlabeled
        gene_nn.train()
        gene_ema.register()

        total = 0
        train_total_loss = 0
        train_total_sup_loss = 0
        train_total_unsup_loss = 0
        train_total_accuracy = 0
        train_total_auc = 0.5
        train_total_aupr = 0

        # # eval for once to verify if the checkpoint is loaded correctly
        # if args.resume == True:
        #     eval_dict = self.evaluate(args=args)
        #     print(eval_dict)

        for _, ((x_lb, y_lb), (x_ulb_w1, x_ulb_w2)) in enumerate(zip(l_train_loader,
                                                                            u_train_loader)):
            
            x_lb = x_lb.to(get_device())
            y_lb = y_lb.to(get_device())
            x_ulb_w1 = x_ulb_w1.to(get_device())
            x_ulb_w2 = x_ulb_w2.to(get_device())

            logits_x_lb = gene_nn(x_lb)

            # calculate BN only for the first batch 
            gene_bn_controller.freeze_bn(gene_nn)
            logits_x_ulb_w1 = gene_nn(x_ulb_w1)
            logits_x_ulb_w2 = gene_nn(x_ulb_w2)
            # print("w1: ", logits_x_ulb_w1[0:10])
            # print("w2: ", logits_x_ulb_w2[0:10])
            gene_bn_controller.unfreeze_bn(gene_nn)

            # Calculate Losses.
            unsup_warmup = np.clip(it / (args.unsup_warmup_pos * args.num_train_iter),
                                a_min=0.0, a_max=1.0) # Unsupervised loss weight rampup parameter.
            x_lb_probas = torch.sigmoid(logits_x_lb)[:,0] # Calculate probabilities.
            sup_loss = gene_loss_function(x_lb_probas, y_lb) # Supervised loss function
            unsup_loss = consistency_loss(logits_x_ulb_w1,
                                            logits_x_ulb_w2) # Unsupervised loss function
            # print("unsupervised loss: {}".format(unsup_loss))
            total_loss = sup_loss + args.ulb_loss_ratio * unsup_loss * unsup_warmup
            accuracy, auc, aupr = performance_eval(x_lb_probas, y_lb) # Performance evaluation

            total_loss.backward()
            gene_optimizer.step()
            gene_ema.update()
            gene_nn.zero_grad()

            total += 1
            train_total_loss += total_loss.item()
            train_total_sup_loss += sup_loss.item()
            train_total_unsup_loss += unsup_loss.item()
            train_total_accuracy += accuracy.item()
            train_total_auc += auc.item()
            train_total_aupr += aupr.item()
            it += 1
        
        train_mean_total_loss = train_total_loss / total
        train_mean_sup_loss = train_total_sup_loss / total
        train_mean_unsup_loss = train_total_unsup_loss / total
        train_mean_accuracy = train_total_accuracy / total
        train_mean_auc = train_total_auc / total
        train_mean_aupr = train_total_aupr / total

        # Performance evaluation with test data.
        total = 0
        test_total_loss = 0
        test_total_accuracy = 0
        test_total_auc = 0.5
        test_total_aupr = 0
        gene_nn.eval()
        gene_ema.apply_shadow()
        
        with torch.no_grad():
            for _, (x_lb, y_lb) in enumerate(test_loader):

                x_lb = x_lb.to(get_device())
                y_lb = y_lb.to(get_device())

                logits_x_lb = gene_nn(x_lb) # Predict
                x_lb_probas = torch.sigmoid(logits_x_lb)[:,0] # Calculate probabilities.
                sup_loss = gene_loss_function(x_lb_probas, y_lb) # loss
                accuracy, auc, aupr = performance_eval(x_lb_probas, y_lb) # Performance evaluation

                total += 1
                test_total_loss += sup_loss.item()
                test_total_accuracy += accuracy.item()
                test_total_auc += auc.item()
                test_total_aupr += aupr.item()

            test_mean_total_loss = test_total_loss / total
            test_mean_accuracy = test_total_accuracy / total
            test_mean_auc = test_total_auc / total
            test_mean_aupr = test_total_aupr / total

        print('Epoch {:d}/{:d}, \
TrainTotal: {:.3f}, TrainSup: {:.3f}, TrainUnsup: {:.3f}, TrainAcc: {:.3f}, TrainAuc: {:.3f}, TrainAupr: {:.3f}, \
TestTotal: {:.3f}, TestAuc: {:.3f}, TestAupr: {:.3f}'.format(\
            epoch+1, args.gene_epochs, 
            train_mean_total_loss, train_mean_sup_loss, train_mean_unsup_loss, train_mean_accuracy, train_mean_auc, train_mean_aupr,
            test_mean_total_loss, test_mean_accuracy, test_mean_auc, test_mean_aupr))
        
        # Save trained results to file
        with open( f"{make_output_directory_path(args)}/{args.gene_nn_train_results}", 'a+') as wf:
            wf.write('{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
            epoch+1, 
            train_mean_total_loss, train_mean_sup_loss, train_mean_unsup_loss, train_mean_accuracy, train_mean_auc, train_mean_aupr,
            test_mean_total_loss, test_mean_accuracy, test_mean_auc, test_mean_aupr))

        gene_ema.restore()
        gene_nn.train()

        early_stopping(test_mean_total_loss, gene_nn) # If best model is obtained, save the model.
        if early_stopping.early_stop:
            # When loss values increase repeatedly, end the training.
            break
        
    # # Save trained GeneVAE
    # if early_stopping.early_stop == False:
    #     gene_nn.save_model( make_output_directory_path(args) + args.saved_gene_nn + '.pkl')
    #     print('Trained ConcatNN is saved in {}'.format( make_output_directory_path(args) + args.saved_gene_nn + '.pkl' ))

    return gene_nn

# ============================================================================







