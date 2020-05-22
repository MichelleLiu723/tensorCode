#!/usr/bin/env python
import argparse
import torch
import numpy as np
import tensornetwork as tn
import matplotlib.pyplot as plt

import core_code as cc
import randomwalk as rw
import randomsearch as rs
np.random.seed(10)
torch.manual_seed(10)

def main(args):
    num_train = args.ntrain
    num_val = args.nval
    input_dims = [7,7,7,7,7]
    goal_tn = torch.load(args.path)  
    base_tn = cc.random_tn(input_dims, rank=1)
    base_tn = cc.make_trainable(base_tn)
    loss_fun = cc.regression_loss
    train_data = cc.generate_regression_data(goal_tn, num_train, noise=1e-6)
    val_data   = cc.generate_regression_data(goal_tn, num_val,   noise=1e-6)
    
    
    best_network, first_loss, best_loss, loss_record, loss_hist, param_count, d_loss_hist = rw.randomwalk_optim(base_tn, 
                                                                                            train_data, 
                                                                                            loss_fun, 
                                                                                            val_data=val_data,
                                                                                            other_args={
                                                                                                'dhist':True,
                                                                                                'optim':'RMSprop',
                                                                                                'max_iter':args.maxiter,
                                                                                                'epochs':None,  # early stopping
                                                                                                'lr':0.001})

    plt.figure(figsize=(4, 3))
    plt.plot(loss_hist[0])
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.savefig('./figures/'+args.path+'_randomwalk'+'_trainloss'+'_.pdf', bbox_inches = 'tight')
    
    plt.figure(figsize=(4, 3))
    plt.plot(param_count, d_loss_hist[0])
    plt.xlabel('Number of parameters')
    plt.ylabel('Training loss')
    plt.savefig('./figures/'+args.path+'_randomwalk'+'_trainloss_numparam'+'_.pdf', bbox_inches = 'tight')
    
    plt.figure(figsize=(4, 3))
    plt.plot(param_count, loss_record)
    plt.xlabel('Number of parameters')
    plt.ylabel('Validation loss')
    plt.savefig('./figures/'+args.path+'_randomwalk'+'_valloss_numparam'+'_.pdf', bbox_inches = 'tight')
    
    ### TODO: Add greedy

    ### random search
    best_network, first_loss, best_loss, loss_record, param_count, d_loss_hist = rs.randomsearch_optim(base_tn, 
                                                                                            train_data, 
                                                                                            loss_fun, 
                                                                                            val_data=val_data,
                                                                                            other_args={
                                                                                                'dhist':True,
                                                                                                'optim':'RMSprop',
                                                                                                'max_iter':args.maxiter,
                                                                                                'epochs':None,  # early stopping
                                                                                                'lr':0.001})
    
    plt.figure(figsize=(4, 3))
    plt.plot(param_count, d_loss_hist[0])
    plt.xlabel('Number of parameters')
    plt.ylabel('Training loss')
    plt.savefig('./figures/'+args.path+'_randomsearch'+'_trainloss_numparam'+'_.pdf', bbox_inches = 'tight')
    
    plt.figure(figsize=(4, 3))
    plt.plot(param_count, loss_record)
    plt.xlabel('Number of parameters')
    plt.ylabel('Validation loss')
    plt.savefig('./figures/'+args.path+'_randomsearch'+'_valloss_numparam'+'_.pdf', bbox_inches = 'tight')  




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',    required=True, type=str, help="path to goal TN")
    parser.add_argument('--ntrain',  required=False, type=int, default=50000, help="number of train data points")
    parser.add_argument('--nval',    required=False, type=int, default=5000,  help="number of validation data points")
    parser.add_argument('--maxiter', required=False, type=int, default=20,    help="maximum number of discrete optimization")
    args = parser.parse_args()
    main(args)