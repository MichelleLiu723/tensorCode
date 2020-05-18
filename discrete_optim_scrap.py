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
def generate_stop_cond(in_dims):
    """
    Example code for how you can produce a stop_cond function that stops
    optimization when the number of parameters in the TN exceeds the 
    number of parameters needed to specify the dense tensor itself
    """
    # Number of elements in the dense tensor with input dimensions in_dims
    max_params = torch.prod(torch.tensor(in_dims))

    # The stop_cond function which is output by generate_stop_cond
    def stop_cond(tensor_list):
        return cc.num_params(tensor_list) >= max_params

    return stop_cond

def discrete_optim_template(tensor_list, train_data, loss_fun, 
                            val_data=None, other_args=dict()):
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
    dhist  = other_args['dhist']  if 'dhist'  in other_args else False
    loss_rec, first_loss, best_loss, best_network = [], None, None, None
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

    # Define a function giving the stop condition for the discrete 
    # optimization procedure. I'm using a simple example here which could
    # work for greedy or random walk searches, but for other optimization
    # methods this could be trivial
    stop_cond = generate_stop_cond(cc.get_indims(tensor_list))

###########***************where are you increasing the rank from line 123 to 137?*****
    # Iteratively increment ranks of tensor_list and train via
    # continuous_optim, at each stage using a search procedure to 
    # test out different ranks before choosing just one to increase
    stage = 0
    better_network, better_loss = tensor_list, 1e10
    while not stop_cond(better_network):
        if first_loss is None:
            # Record initial loss of TN model 
            first_args = other_args
            first_args["print"] = first_args["hist"] = False
            _, first_loss, _ = cc.continuous_optim(tensor_list, train_data, 
                                    loss_fun, epochs=1, val_data=val_data,
                                    other_args=first_args)
            m_print("Initial model has TN ranks")
            if dprint: cc.print_ranks(tensor_list)
            m_print(f"Initial loss is {first_loss:.3f}")
            continue
        m_print(f"STAGE {stage}")

##################################line 139 onward are new added acode#############
        # Try out training different network ranks and assign network
        # with best ranks to better_network
        # 
        # TODO: This is your part to write! Use new variables for the loss
        #       and network being tried out, with the network being
        #       initialized from better_network. When you've found a better
        #       TN, make sure to update better_network and better_loss to
        #       equal the parameters and loss of that TN.
        # At some point this code will call the continuous optimization
        # loop, in which case you can use the following command:
        # 
        # trained_tn, init_loss, final_loss = cc.continuous_optim(my_tn, 
        #                                         train_data, loss_fun, 
        #                                         epochs=epochs, 
        #                                         val_data=val_data,
        #                                         other_args=other_args)

        paramKey = -1
        #best_loss = float('inf')
        prev_loss = float('inf')
        max_params = torch.prod(torch.tensor((cc.get_indims(train_data))))  #train data is the target tensor
        currentNetwork  = cc.copy_network(tensor_list) #temp
        initialNetwork  = cc.copy_network(tensor_list) #example_tn
        loss_record = []
        for k in range(5):   
            if paramKey == 1:
                break
            for i in range(len(currentNetwork)):
                if paramKey == 1:
                    break
                for j in range(i, len(currentNetwork)):
                    if paramKey ==1:
                        break
                    if i==j:
                        continue
                    print('k = ', k, 'i =', i, 'j = ', j)
                    stop_cond = generate_stop_cond(cc.get_indims(currentNetwork)) #***********need to add param=-1
                    if stop_cond == True:  #i.e. break if the number of parameters in trained tensor exceeds number of param. in target tensor
                        break
                    #solve continuos optimization part, train_data = target_tensor
                    [currentNetwork, first_loss, current_loss] = cc.continuous_optim(currentNetwork, train_data, loss_fun, epochs=20, val_data=None, other_args=dict())
                    if prev_loss > current_loss:
                        prev_loss = current_loss
                        better_network = currentNetwork
                        numParam = cc.num_params(better_network)
                        better_loss = current_loss
                    else:
                        print('prev_loss < current_loss so no rank update at this iteration k =', k, 'i = ', i, 'j = ', j)
                        print('prev_loss = ', prev_loss)
                    #reset currentNetwork to contuniue with greedy at another point
                    currentNetwork = cc.copy_network(initialNetwork)
                    #increase rank along a chosen dimension
                    currentNetwork = cc.increase_rank(currentNetwork,i, j, 1, 1e-6)
            #upate parameters
            currentNetwork = cc.copy_network(better_network)
            #update current point to the new point (i.e. better_network) that gave lower loss
            initialNetwork  = cc.copy_network(better_network)
            loss_record.append(better_loss)
        return best_network, first_loss, better_loss #, loss_record

#for testing

if __name__ == '__main__':
    #Target tensor is a chain
    #Tensor decomposition
    d0 = 4
    d1 = 4
    d2 = 4
    d3 = 4
    d4 = 4
    d5 = 4
    r12 = 2
    r23 = 3
    r34 = 6
    r45 = 5
    r56 = 4
    input_dims = [d0, d1, d2, d3, d4, d5]
    rank_list = [[r12, 1, 1, 1, 1], 
                     [r23,1, 1, 1], 
                         [r34, 1, 1],
                              [r45, 1],
                                   [r56]]
    
    loss_fun = cc.tensor_recovery_loss
    base_tn = cc.random_tn(input_dims, rank=1)
    goal_tn = cc.random_tn(input_dims, rank=rank_list)
    base_tn = cc.make_trainable(base_tn)
    trained_tn, init_loss, better_loss = discrete_optim_template(base_tn, 
                                                        goal_tn, loss_fun)
    print('better loss = ', better_loss)


