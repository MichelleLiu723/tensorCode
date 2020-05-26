#!/usr/bin/env python
import random
from functools import partial, reduce
from copy import deepcopy

import torch
import numpy as np
import tensornetwork as tn

import core_code as cc
"""
### NOTES ON TENSOR NETWORK FORMAT ###

A tensor network is specified as a list of tensors, each of whose shape is
formatted in a specific way. In particular, for a network with n tensor
cores, the shape will be:

tensor_i.shape = (r_1, r_2, ..., r_i, ..., r_n),

where r_j gives the tensor network rank (TN-rank) connecting cores i and j
when j != i, while r_i gives the dimension of the input to core i. This 
implies that tensor_i.shape[j] == tensor_j.shape[i], and stacking the 
shapes of all the tensors in order gives the adjacency matrix of the 
network (ignoring the diagonals).

On occasion, the ranks are specified in the following triangular format:
     [[r_{1,2}, r_{1,3}, ..., r_{1,n}], [r_{2,3}, ..., r_{2,n}], ...
      ..., [r_{n-2,n-1}, r_{n-2,n}], [r_{n-1,n}]]
"""
def training(tensor_list, train_data, loss_fun, epochs,
             val_data=None, other_args=dict()):
    """Run continuous optimization with adaptive learning rate"""
    args_copy = deepcopy(other_args)
    hist = None
    while hist is None or (hist[0][0] < hist[0][-1]):
        args_copy["lr"] /= 2
        currentNetwork, first_loss, current_loss, hist = cc.continuous_optim(
                                                    tensor_list, train_data, 
                                                    loss_fun, val_data=val_data, 
                                                    epochs=epochs, 
                                                    other_args=args_copy)
    
    return currentNetwork, first_loss, current_loss, hist

def randomwalk_optim(tensor_list, train_data, loss_fun, 
                            val_data=None, other_args=dict()):
    """
    Train a tensor network using discrete optimization over TN ranks

    Args:
        tensor_list: List of tensors encoding the network being trained
        train_data:  The data used to train the network
        loss_fun:    Scalar-valued loss function of the type 
                        tens_list, data -> scalar_loss
                     (This depends on the task being learned)
        val_data:    The data used for validation, which can be used to
                     for early stopping within continuous optimization
                     calls within discrete optimization loop
        other_args:  Dictionary of other arguments for the optimization, 
                     with some options below (feel free to add more)

                        epochs:      Number of epochs for 
                                     continuous optimization     (default=10)
                        max_iter:    Maximum number of iterations
                                     for random search           (default=10)
                        optim:       Choice of Pytorch optimizer (default='SGD')
                        lr:          Learning rate for optimizer (default=1e-3)
                        bsize:       Minibatch size for training (default=100)
                        reps:        Number of times to repeat 
                                     training data per epoch     (default=1)
                        cprint:      Whether to print info from
                                     continuous optimization     (default=True)
                        dprint:      Whether to print info from
                                     discrete optimization       (default=True)
                        dhist:       Whether to return losses
                                     from intermediate TNs       (default=False)
                        loss_threshold: Threshold for discrete 
                                        optimization             (default=1e-5)
    
    Returns:
        better_list: List of tensors with same length as tensor_list, but
                     having been optimized using the discrete optimization
                     algorithm. The TN ranks of better_list will be larger
                     than those of tensor_list.
        first_loss:  Initial loss of the model on the validation set, 
                     before any training. If no val set is provided, the
                     first training loss is instead returned
        best_loss:   The value of the validation/training loss for the
                     model output as better_list
        loss_record: If dhist=True in other_args, all values of best_loss
                     associated with intermediate optimized TNs will be
                     returned as a PyTorch vector, with loss_record[0]
                     giving the initial loss of the model, and
                     loss_record[-1] equal to best_loss value returned
                     by discrete_optim.
        loss_hist:   List of losses for every continous and discrete step.
                     Each element is a tuple of the form (train_loss, val_loss)
        d_loss_hist: List of losses at end of each discrete optimization step
                     Each element is a tuple of the form (train_loss, val_loss)
        param_count: List of number of parameters.
    """
    # Check input and initialize local record variables
    epochs  = other_args['epochs'] if 'epochs' in other_args else 10
    max_iter  = other_args['max_iter'] if 'max_iter' in other_args else 10
    dprint  = other_args['dprint'] if 'dprint' in other_args else True
    dhist  = other_args['dhist']  if 'dhist'  in other_args else False
    loss_threshold = other_args['loss_threshold']  if 'loss_threshold'  in other_args else 1e-5
    loss_hist, first_loss, best_loss, best_network = ([],[]), None, None, None
    d_loss_hist = ([], [])
    
    # Keep track of loss in continous optimization
    other_args['hist'] = True
    
    # Record number of parameters for each step
    param_count = []
    
    if dhist: loss_record = []    # (train_record, val_record)

    # Function to maybe print, conditioned on `dprint`
    m_print = lambda s: print(s) if dprint else None

    # Function to record loss information
    def record_loss(new_loss, new_network, new_loss_hist):
        # Load record variables from outer scope
        nonlocal first_loss, best_loss, best_network, loss_hist, d_loss_hist
        
        # Add full loss history
        loss_hist[0].extend(new_loss_hist[0].tolist())
        loss_hist[1].extend(new_loss_hist[1].tolist())
        
        # Add discrete loss history
        d_loss_hist[0].append(new_loss_hist[0][-1])
        d_loss_hist[1].append(new_loss_hist[1][-1])
        
        # Track number of parameters of new_network
        nonlocal param_count
        param_count.append(cc.num_params(new_network))
        
        # Check for best loss
        if best_loss is None or new_loss < best_loss:
            best_loss, best_network = new_loss, new_network

        # Add new loss to our loss record
        if not dhist: return
        nonlocal loss_record
        loss_record.append(new_loss)

    # Copy tensor_list so the original is unchanged
    tensor_list = cc.copy_network(tensor_list)

    # Define a function giving the stop condition for the discrete 
    # optimization procedure. I'm using a simple example here which could
    # work for greedy or random walk searches, but for other optimization
    # methods this could be trivial
    stop_cond = lambda loss: loss < loss_threshold

    # Iteratively increment ranks of tensor_list and train via
    # continuous_optim, at each stage using a search procedure to 
    # test out different ranks before choosing just one to increase
    stage = 0
    better_network, better_loss = tensor_list, 1e10
    while not stop_cond(better_loss) and stage < max_iter:
        stage += 1
        better_network, ـ, better_loss, new_loss_hist = training(
            better_network,
            train_data, loss_fun,
            epochs=epochs,
            val_data=val_data,
            other_args=other_args)
        
        # Randomly pick 2 tensors to increase the bond dimension between them
        idx = torch.randperm(len(better_network))[:2]
        
        # Create new TN with increased rank
        better_network = cc.increase_rank(better_network, vertex1=idx[0], vertex2=idx[1])
        better_network = cc.make_trainable(better_network)

        # Record the loss associated with the best network from this 
        # discrete optimization loop
        record_loss(better_loss, better_network, new_loss_hist)
        m_print(f"STAGE {stage}")

    if dhist:
        loss_record = tuple(torch.tensor(fr) for fr in loss_record)
        return best_network, first_loss, best_loss, loss_record, loss_hist, param_count, d_loss_hist
    else:
        return best_network, first_loss, best_loss

if __name__ == '__main__':
    np.random.seed(10)
    torch.manual_seed(10)
    num_train = 50000
    num_val = 5000
    goal_tn = torch.load('tt_cores_5.pt')
    input_dims = [7,7,7,7,7]
    base_tn = cc.random_tn(input_dims, rank=1)
    base_tn = cc.make_trainable(base_tn)
    train_data = cc.generate_regression_data(goal_tn, num_train, noise=1e-6)
    val_data   = cc.generate_regression_data(goal_tn, num_val,   noise=1e-6)
    loss_fun = cc.regression_loss
    
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    lr_scheduler = lambda optimizer: ReduceLROnPlateau(optimizer, mode='min', factor=1e-10, patience=200, verbose=True,threshold=1e-7)
    
    best_network, first_loss, best_loss, loss_record, loss_hist, param_count, d_loss_hist = randomwalk_optim(base_tn, 
                                                                                            train_data, 
                                                                                            loss_fun, 
                                                                                            val_data=None,
                                                                                            other_args={
                                                                                                'dhist':True,
                                                                                                'optim':'RMSprop',
                                                                                                'max_iter':20,
                                                                                                'epochs':1e10,  # early stopping
                                                                                                'cvg_threshold':1e-10, 
                                                                                                'lr_scheduler':lr_scheduler, 
                                                                                                'lr':0.01})