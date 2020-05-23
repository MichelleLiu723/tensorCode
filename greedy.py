#!/usr/bin/env python
import random
from functools import partial, reduce

import torch
import numpy as np
import tensornetwork as tn

import core_code as cc
from copy import deepcopy

def training(tensor_list, initial_epochs, train_data, 
            loss_fun, val_data, epochs, other_args):
    """
    This function run the continuous optimization routine with a small tweak: if there is no improvement of the loss in the 
    first [initial_epochs] epochs, the learning rate is reduced by a factor of 0.5 and optimization is restarted from the beginning,
    All arguments are the same as cc.continuous_optim except for [initial_epochs].
    """

    if initial_epochs is None:
        return cc.continuous_optim(currentNetwork, train_data, 
            loss_fun, val_data=val_data, epochs=epochs, other_args=args)

    args = deepcopy(other_args)
    args["hist"] = True
    current_network_optimizer_state = other_args["optimizer_state"] if "optimizer_state" in args else {}
    args["save_optimizer_state"] = True
    args["optimizer_state"] = current_network_optimizer_state
    currentNetwork = cc.copy_network(tensor_list)
    remaining_epochs = epochs - initial_epochs if epochs else None
    hist = None
    while hist is None or (hist[0][0] < hist[0][-1]):
        if hist:
            if "load_optimizer_state" in args and args["load_optimizer_state"]:
                args["load_optimizer_state"]["optimizer_state"]['param_groups'][0]['lr'] /= 2
                lr = args["load_optimizer_state"]["optimizer_state"]['param_groups'][0]['lr']
            else:
                args["lr"] /= 2
                lr = args["lr"]
            print(f"\n[!] No progress in first {initial_epochs} epochs, starting again with smaller learning rate ({lr})")
            currentNetwork = cc.copy_network(tensor_list)
        [currentNetwork, first_loss, current_loss, hist] = cc.continuous_optim(currentNetwork, train_data, 
            loss_fun, val_data=val_data, epochs=initial_epochs, other_args=args)

    args["load_optimizer_state"] = current_network_optimizer_state
    [currentNetwork, first_loss, current_loss, hist] = cc.continuous_optim(currentNetwork, train_data, 
            loss_fun, val_data=val_data, epochs=remaining_epochs, other_args=args)
    
    return [currentNetwork, first_loss, current_loss, hist] if "hist" in other_args else [currentNetwork, first_loss, current_loss]




def greedy_optim(tensor_list, train_data, loss_fun, 
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
                        max_iter: Maximum number of iterations
                                  for greedy search           (default=10)
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
                        initial_epochs: Number of epochs after which the 
                        learning rate is reduced and optimization is restarted
                        if there is no improvement in the loss.
                                                            (default=None)
    
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
    max_iter  = other_args['max_iter'] if 'max_iter' in other_args else 10
    dprint  = other_args['dprint'] if 'dprint' in other_args else True
    cprint  = other_args['cprint'] if 'cprint' in other_args else True
    dhist  = other_args['dhist']  if 'dhist'  in other_args else False
    search_epochs  = other_args['search_epochs']  if 'search_epochs'  in other_args else None
    loss_threshold  = other_args['loss_threshold']  if 'loss_threshold'  in other_args else 1e-5
    initial_epochs  = other_args['initial_epochs'] if 'initial_epochs' in other_args else None
    stop_cond = lambda loss: loss < loss_threshold
    loss_threshold = other_args['loss_threshold']  if 'loss_threshold'  in other_args else 1e-4 # 1e-5
    loss_hist, first_loss, best_loss, best_network = ([],[]), None, None, None
    d_loss_hist = ([], [])

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
        # loss_hist[0].extend(new_loss_hist[0].tolist())
        # loss_hist[1].extend(new_loss_hist[1].tolist())
       # loss_hist[0].extend(new_loss_hist.tolist())
       # loss_hist[1].extend(new_loss_hist.tolist())
        
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
    loss_record, best_loss, best_network = [], np.infty, None

    while not stop_cond(best_loss) and stage < max_iter:
        stage += 1
        if best_loss is np.infty: # first continuous optimization
            tensor_list, first_loss, best_loss = training(
                tensor_list, initial_epochs, train_data, loss_fun, epochs=epochs, 
                val_data=val_data,other_args=other_args)
            m_print("Initial model has TN ranks")
            if dprint: cc.print_ranks(tensor_list)
            m_print(f"Initial loss is {best_loss:.7f}")

            initialNetwork = cc.copy_network(tensor_list) 
            best_network = cc.copy_network(tensor_list) 
            loss_record += [first_loss,best_loss]


        m_print(f"\n\n**** Discrete optimization - iteration {stage} ****\n\n\n")  
        best_search_loss = best_loss
        for i in range(len(initialNetwork)):
            for j in range(i+1, len(initialNetwork)):
                currentNetwork = cc.copy_network(initialNetwork)
                #increase rank along a chosen dimension
                currentNetwork = cc.increase_rank(currentNetwork,i, j, 1, 1e-6)
                currentNetwork = cc.make_trainable(currentNetwork)
                print('\ntesting rank increment for i =', i, 'j = ', j)
                if search_epochs: # we d only a few epochs to identify the most promising rank update

                    # function to zero out the gradient of all entries except for the new slices
                    def grad_masking_function(tensor_list):
                        nonlocal i,j
                        for k in range(len(tensor_list)):
                            if k == i:
                                tensor_list[i].grad.permute([j]+list(range(0,j))+list(range(j+1,len(currentNetwork))))[:-1,:,...] *= 0
                            elif k == j:
                                tensor_list[j].grad.permute([i]+list(range(0,i))+list(range(i+1,len(currentNetwork))))[:-1,:,...] *= 0
                            else:
                                tensor_list[k].grad *= 0

                    # we first optimize only the new slices for a few epochs
                    print("optimize new slices for a few epochs")
                    search_args = dict(other_args)
                    search_args["hist"] = True
                    current_network_optimizer_state = {}
                    search_args["save_optimizer_state"] = True
                    search_args["optimizer_state"] = current_network_optimizer_state
                    search_args["grad_masking_function"] = grad_masking_function
                    [currentNetwork, first_loss, current_loss, hist] = training(currentNetwork, initial_epochs, train_data, 
                        loss_fun, val_data=val_data, epochs=search_epochs, other_args=search_args)
                    first_loss = hist[0][0]

                    # We then optimize all parameters for a few epochs
                    print("\noptimize all parameters for a few epochs")
                    search_args["grad_masking_function"] = None
                    search_args["load_optimizer_state"] = dict(current_network_optimizer_state)
                    [currentNetwork, first_loss, current_loss, hist] = training(currentNetwork, initial_epochs, train_data, 
                        loss_fun, val_data=val_data, epochs=search_epochs , 
                        other_args=search_args)
                    search_args["load_optimizer_state"] = None

                else: # we fully optimize the network in the search phase
                    [currentNetwork,  first_loss, current_loss] = cc.continuous_optim(currentNetwork, train_data, 
                        loss_fun, val_data=val_data, epochs=epochs, 
                        other_args=other_args)
                # Record the loss associated with the best network from this 
                # discrete optimization loop 
               # record_loss(first_loss, currentNetwork ,current_loss)
               # m_print(f"STAGE {stage}")

                m_print(f"\nCurrent loss is {current_loss:.7f}    Best loss from previous discrete optim is {best_loss}")
                if best_search_loss > current_loss:
                    best_search_loss = current_loss
                    best_network = currentNetwork
                    best_network_optimizer_state = deepcopy(current_network_optimizer_state)
                    print('-> best rank update so far:', i,j)
        best_loss = best_search_loss
        # train network to convergence for the best rank increment (if search_epochs is set, 
        # otherwise the best network is already trained to convergence / max_epochs)
        if search_epochs:
            print('\ntraining best network until max_epochs/convergence...')
            other_args["load_optimizer_state"] = best_network_optimizer_state
            current_network_optimizer_state = {}
            other_args["save_optimizer_state"] = True
            other_args["optimizer_state"] = current_network_optimizer_state
            [best_network, first_loss, best_loss] = training(best_network, initial_epochs, train_data, 
                    loss_fun, val_data=val_data, epochs=epochs, 
                    other_args=other_args)
            other_args["load_optimizer_state"] = None

        initialNetwork  = cc.copy_network(best_network)
        loss_record.append((stage, cc.num_params(best_network), float(best_loss)))
        print('\nbest TN:')
        cc.print_ranks(best_network)
        print('number of params:',cc.num_params(best_network))
        print(loss_record)
    if dhist:
        loss_record = tuple(torch.tensor(fr) for fr in loss_record)
        return best_network, first_loss, best_loss, loss_record, param_count, d_loss_hist
    else:
        return best_network, first_loss, best_loss

#for testing

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    # Old target:
    #
    # d0,d1,d2,d3,d4,d5   = 4,4,4,4,4,4
    # r12,r23,r34,r45,r56 =  2,3,6,5,4
    # input_dims = [d0, d1, d2, d3, d4, d5]
    # rank_list = [[r12, 1, 1, 1, 1], 
    #                  [r23,1, 1, 1], 
    #                      [r34, 1, 1],
    #                           [r45, 1],
    #                                [r56]]

    # New target
    d0,d1,d2,d3,d4   = 7,7,7,7,7
    r12,r23,r34,r45 =  2,3,6,5

    input_dims = [d0, d1, d2, d3, d4]
    # rank_list = [[r12, 1, 1, 1], 
    #                  [r23,1, 1], 
    #                      [r34, 1],
    #                           [r45]]
    
    loss_fun = cc.tensor_recovery_loss
    base_tn = cc.random_tn(input_dims, rank=1)
    
    # Initialize the first tensor network close to zero
    for i in range(len(base_tn)):
        base_tn[i] /= 10
    base_tn = cc.make_trainable(base_tn)
    goal_tn = torch.load('tt_cores_5.pt')
    print('target tensor network number of params: ', cc.num_params(goal_tn))
    print('number of params for full target tensor:', np.prod(input_dims))
    print('target tensor norm:', cc.l2_norm(goal_tn))


    from torch.optim.lr_scheduler import ReduceLROnPlateau

    lr_scheduler = lambda optimizer: ReduceLROnPlateau(optimizer, mode='min', factor=1e-10, patience=200, verbose=True,threshold=1e-7)
    trained_tn, init_loss, better_loss = greedy_optim(base_tn, 
                                                      goal_tn, loss_fun, 
                                                      val_data=goal_tn, #None, 
                                                      other_args={'cprint':True, 'epochs':None, 'max_iter':20, 
                                                                  'lr':0.01, 'optim':'RMSprop', 'search_epochs':80, 
                                                                  'cvg_threshold':1e-10, 'lr_scheduler':lr_scheduler, 
                                                                  'dyn_print':True, 'initial_epochs':10})
 