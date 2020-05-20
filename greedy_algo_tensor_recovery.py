#!/usr/bin/env python
import random
from functools import partial, reduce

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


def discrete_optim_template(tensor_list, train_data, loss_fun, 
                            val_data=None, other_args=dict(),max_iter=None):
    """
    Train a tensor network using discrete optimization over TN ranks
    Args:
        tensor_list: List of tensors encoding the network being trained
        train_data:  The data used to train the network(target Tensor)
        loss_fun:    Scalar-valued loss function of the type 
                        tens_list, data -> scalar_loss
                     (This depends on the task being learned)
        val_data:    The data used for validation, which can be used to
                     for early stopping within continuous optimization
                     calls within discrete optimization loop
        other_args:  Dictionary of other arguments for the optimization, 
                     with some options below (feel free to add more)
                        epochs: Number of epochs for 
                                continuous optimization     (default=10)
                        optim:  Choice of Pytorch optimizer (default='SGD')
                        lr:     Learning rate for optimizer (default=1e-3)
                        bsize:  Minibatch size for training (default=100)
                        reps:   Number of times to repeat 
                                training data per epoch     (default=1)
                        cprint: Whether to print info from
                                continuous optimization     (default=True)
                        dprint: Whether to print info from
                                discrete optimization       (default=True)
                        dhist:  Whether to return losses
                                from intermediate TNs       (default=False)
                        search_epochs: Number of epochs to use to identify the
                                best rank 1 update. If None, the epochs argument
                                is used.                    (default=None)
                        loss_threshold: if loss gets below this threshold, 
                        discrete optimization is stopped
                                                            (default=1e-5)
    
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
    """
    # Check input and initialize local record variables
    epochs  = other_args['epochs'] if 'epochs' in other_args else 10
    dprint  = other_args['dprint'] if 'dprint' in other_args else True
    cprint  = other_args['cprint'] if 'cprint' in other_args else True
    dhist  = other_args['dhist']  if 'dhist'  in other_args else False
    search_epochs  = other_args['search_epochs']  if 'search_epochs'  in other_args else None
    loss_threshold  = other_args['loss_threshold']  if 'loss_threshold'  in other_args else 1e-5
    stop_cond = lambda loss: loss < loss_threshold


    if dhist: loss_record = []    # (train_record, val_record)

    # Function to maybe print, conditioned on `dprint`
    m_print = lambda s: print(s) if dprint else None

    # Function to record loss information
    def record_loss(new_loss, new_network):
        # Load record variables from outer scope
        nonlocal loss_rec, first_loss, best_loss, best_network

        # Check for best loss
        if best_loss is None or new_loss < best_loss:
            best_loss, best_network = new_loss, new_network

        # Add new loss to our loss record
        if not dhist: return
        nonlocal loss_record
        loss_record.append(new_loss)

    # Copy tensor_list so the original is unchanged
    tensor_list = cc.copy_network(tensor_list)


    # Iteratively increment ranks of tensor_list and train via
    # continuous_optim, at each stage using a search procedure to 
    # test out different ranks before choosing just one to increase
    iter = 0
    loss_rec, prev_loss, best_loss, best_network = [], None, None, None
    best_network, better_loss = tensor_list, np.infty


    while (not stop_cond(better_loss)) or (max_iter and iter > max_iter):
        iter += 1

        if prev_loss is None:
            # Record initial loss of TN model 
            tensor_list, _, prev_loss = cc.continuous_optim(tensor_list, train_data, 
                                    loss_fun, epochs=epochs, val_data=val_data,
                                    other_args=other_args)
            m_print("Initial model has TN ranks")
            if dprint: cc.print_ranks(tensor_list)
            m_print(f"Initial loss is {prev_loss:.7f}")

            initialNetwork  = cc.copy_network(tensor_list) #example_tn
            

##################################line 139 onward are new added acode#############
        # Try out training different network ranks and assign network
        # with best ranks to best_network
        # 
        # TODO: This is your part to write! Use new variables for the loss
        #       and network being tried out, with the network being
        #       initialized from best_network. When you've found a better
        #       TN, make sure to update best_network and better_loss to
        #       equal the parameters and loss of that TN.
        # At some point this code will call the continuous optimization
        # loop, in which case you can use the following command:
        # 
        # trained_tn, init_loss, final_loss = cc.continuous_optim(my_tn, 
        #                                         train_data, loss_fun, 
        #                                         epochs=epochs, 
        #                                         val_data=val_data,
        #                                         other_args=other_args)
        #best_loss = float('inf')

        loss_record = [] 

        m_print(f"**** Discrete optimization - iteration {iter} ****")   
        for i in range(len(initialNetwork)):
            for j in range(i+1, len(initialNetwork)):
                currentNetwork = cc.copy_network(initialNetwork)
                #increase rank along a chosen dimension
                currentNetwork = cc.increase_rank(currentNetwork,i, j, 1, 1e-10)

                print('testing rank increment for i =', i, 'j = ', j)
                if search_epochs:
                    search_args = dict(other_args)
                    [currentNetwork, first_loss, current_loss] = cc.continuous_optim(currentNetwork, train_data, 
                        loss_fun, val_data=val_data, epochs=search_epochs , 
                        other_args=search_args)
                else:
                    [currentNetwork, first_loss, current_loss] = cc.continuous_optim(currentNetwork, train_data, 
                        loss_fun, val_data=val_data, epochs=epochs, 
                        other_args=other_args)
                
                m_print(f"\nCurrent loss is {current_loss:.7f}")
                if prev_loss > current_loss:
                    prev_loss = current_loss
                    best_network = currentNetwork
                    numParam = cc.num_params(best_network)
                    better_loss = current_loss
                    print('best rank update so far:', i,j)
        
        # train network to convergence for the best rank increment (if search_epochs is set, 
        # otherwise the best network is already trained to convergence / max_epochs)
        if search_epochs:
            print('training best network until max_epochs/convergence..')
            [best_network, first_loss, better_loss] = cc.continuous_optim(best_network, train_data, 
                    loss_fun, val_data=val_data, epochs=epochs, 
                    other_args=other_args)
        currentNetwork = cc.copy_network(best_network)
        #update current point to the new point (i.e. best_network) that gave lower loss
        initialNetwork  = cc.copy_network(best_network)
        loss_record.append(better_loss)
        print('\nbest TN:')
        cc.print_ranks(best_network)
        print('number of params:',cc.num_params(best_network))

    return best_network, first_loss, better_loss #, loss_record

#for testing

if __name__ == '__main__':
    #torch.manual_seed(0)
    #Target tensor is a chain
    #Tensor decomposition
    #torch.manual_seed(0)
    d0 = 4
    d1 = 4
    d2 = 4
    d3 = 4
    d4 = 4
    d5 = 4
    r12 = 2
    r23 = 3
    r34 = 4
    r45 = 5
    r56 = 2
    input_dims = [d0, d1, d2, d3, d4, d5]
    rank_list = [[r12, 1, 1, 1, 1], 
                     [r23,1, 1, 1], 
                         [r34, 1, 1],
                              [r45, 1],
                                   [r56]]
    
    loss_fun = cc.tensor_recovery_loss
    base_tn = cc.random_tn(input_dims, rank=1)
    # Initialize the first tensor network close to zero
    for i in range(len(base_tn)):
        base_tn[i] /= 10
    goal_tn = cc.random_tn(input_dims, rank=rank_list)
    base_tn = cc.make_trainable(base_tn)
    print('target tensor number of params:', cc.num_params(goal_tn))
    print('target tensor norm:', cc.l2_norm(goal_tn))


    from torch.optim.lr_scheduler import ReduceLROnPlateau

    lr_scheduler = lambda optimizer: ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True,threshold=1e-7)
    trained_tn, init_loss, better_loss = discrete_optim_template(base_tn, 
                                                        goal_tn, loss_fun, 
                                                        val_data=None, 
                                                        other_args={'cprint':True, 'epochs':2000,'lr':0.01, 'optim':'RMSprop',
                                                        'search_epochs':20, 'cvg_threshold':1e-10, 'lr_scheduler':lr_scheduler})

    print('better loss = ', better_loss)


