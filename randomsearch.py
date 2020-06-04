#!/usr/bin/env python
import random
from functools import partial, reduce

import torch
import numpy as np
import tensornetwork as tn

import core_code as cc
from copy import deepcopy

from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import DetectPlateau

def training(tensor_list, initial_epochs, train_data, 
            loss_fun, val_data, epochs, other_args):
    """
    This function run the continuous optimization routine with a small tweak: if there is no improvement of the loss in the 
    first [initial_epochs] epochs, the learning rate is reduced by a factor of 0.5 and optimization is restarted from the beginning,
    All arguments are the same as cc.continuous_optim except for [initial_epochs].
    """

    if initial_epochs is None:
        return cc.continuous_optim(tensor_list, train_data, 
            loss_fun, val_data=val_data, epochs=epochs, other_args=other_args)

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
        [currentNetwork, first_loss, current_loss, best_epoch, hist] = cc.continuous_optim(currentNetwork, train_data, 
            loss_fun, val_data=val_data, epochs=initial_epochs, other_args=args)

    hist_initial = [h[:best_epoch].tolist() for h in hist]
    best_epoch_initial = best_epoch

    args["load_optimizer_state"] = current_network_optimizer_state
    [currentNetwork, first_loss, current_loss, best_epoch, hist] = cc.continuous_optim(currentNetwork, train_data, 
            loss_fun, val_data=val_data, epochs=remaining_epochs, other_args=args)
    
    hist = [h_init + h[:best_epoch].tolist() for h_init,h in zip(hist_initial,hist)]
    best_epoch += best_epoch_initial

    return [currentNetwork, first_loss, current_loss, best_epoch, hist] if "hist" in other_args else [currentNetwork, first_loss, current_loss]


def _make_ranks(ranks):
    """Converts list to triangular ranks structure"""
    n_cores = 1 + (int(5+len(ranks)**0.5))//2
    rank_list = [ranks[i*(i+1)//2:(i+1)*(i+2)//2] for i in range(n_cores-1)]
    return rank_list[::-1]

def _limit_random_tn(input_dims,  rank, max_params):
    """Random TN with max_params (rejection sampling)"""
    new_tn = cc.random_tn(input_dims,  rank)
    if cc.num_params(new_tn)< max_params:
        return new_tn
    else: 
        _limit_random_tn(input_dims,  rank, max_params)

def randomsearch_optim(tensor_list, train_data, loss_fun, 
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
                        epochs: Number of epochs for 
                                continuous optimization     (default=10)
                        max_iter: Maximum number of iterations
                                  for random search         (default=10)
                        optim:  Choice of Pytorch optimizer (default='SGD')
                        lr:     Learning rate for optimizer (default=1e-3)
                        bsize:  Minibatch size for training (default=100)
                        max_rank: Maximum rank to search for  (default=7)
                        reps:   Number of times to repeat 
                                training data per epoch     (default=1)
                        cprint: Whether to print info from
                                continuous optimization     (default=True)
                        dprint: Whether to print info from
                                discrete optimization       (default=True)
                        loss_threshold: if loss gets below this threshold, 
                            discrete optimization is stopped
                                                            (default=1e-5)
                        initial_epochs: Number of epochs after which the 
                            learning rate is reduced and optimization is restarted
                            if there is no improvement in the loss.
                                                            (default=None)
                        stop_on_plateau: a dictionnary containing keys
                                mode  (min/max)
                                patience
                                threshold
                            used to stop continuous optimizaion when plateau
                            is detected                     (default=None)
    
    Returns:
        better_list: List of tensors with same length as tensor_list, but
                     having been optimized using the discrete optimization
                     algorithm. The TN ranks of better_list will be larger
                     than those of tensor_list.
        best_loss:   The value of the validation/training loss for the
                     model output as better_list
        loss_record: If dhist=True in other_args, this records the history of 
                     all losses for discrete and continuous optimization. This
                     is a list of dictionnaries with keys
                           iter,num_params,network,loss,train_loss_hist
                     where
                     iter: iteration of the discrete optimization
                     num_params: number of parameters of best network for this iteration
                     network: list of tensors for the best network for this iteration
                     loss: loss achieved by the best network in this iteration
                     train_loss_hist: history of losses for the continuous optimization
                        for this iteration starting from previous best_network to 
                        the epoch where the new best network was found
                     TODO: add val_lost_hist to the list of keys
    """
    # Check input and initialize local record variables
    epochs  = other_args['epochs'] if 'epochs' in other_args else 10
    max_iter  = other_args['max_iter'] if 'max_iter' in other_args else 10
    max_rank  = other_args['max_rank'] if 'max_rank' in other_args else 7
    max_params = other_args['max_params'] if 'max_params' in other_args else 10000
    dprint  = other_args['dprint'] if 'dprint' in other_args else True
    cprint  = other_args['cprint'] if 'cprint' in other_args else True
    loss_threshold  = other_args['loss_threshold']  if 'loss_threshold'  in other_args else 1e-5
    initial_epochs  = other_args['initial_epochs'] if 'initial_epochs' in other_args else None
    # Keep track of continous optimization history
    other_args['hist'] = True
    
    stop_on_plateau  = other_args['stop_on_plateau'] if 'stop_on_plateau' in other_args else None
    if stop_on_plateau:
        detect_plateau = DetectPlateau(**stop_on_plateau)
        other_args["stop_condition"] = lambda train_loss,val_loss : detect_plateau(train_loss)


    first_loss, best_loss, best_network = ([],[]), None, None
    d_loss_hist = ([], [])

    # Function to maybe print, conditioned on `dprint`
    m_print = lambda s: print(s) if dprint else None



    # Copy tensor_list so the original is unchanged
    tensor_list = cc.copy_network(tensor_list)
    loss_record = [{'iter':-1,'network':tensor_list,
            'num_params':cc.num_params(tensor_list),
            'train_loss_hist':[0]}]  

    # Define a function giving the stop condition for the discrete 
    # optimization procedure. I'm using a simple example here which could
    # work for greedy or random walk searches, but for other optimization
    # methods this could be trivial
    stop_cond = lambda loss: loss < loss_threshold

    input_dims = cc.get_indims(tensor_list)
    n_cores = len(input_dims)
    n_edges = n_cores*(n_cores-1)//2

    # Iteratively increment ranks of tensor_list and train via
    # continuous_optim, at each stage using a search procedure to 
    # test out different ranks before choosing just one to increase
    stage = -1
    best_loss, best_network, best_network_optimizer_state = np.infty, None, None

    while not stop_cond(best_loss) and stage < max_iter:
        stage += 1
        if best_loss is np.infty: # first continuous optimization
            tensor_list, first_loss, best_loss, best_epoch, hist = training(tensor_list, initial_epochs, 
                                    train_data, loss_fun, epochs=epochs, 
                                    val_data=val_data, other_args=other_args)
            m_print("Initial model has TN ranks")
            if dprint: cc.print_ranks(tensor_list)
            m_print(f"Initial loss is {best_loss:.7f}")

            initialNetwork = cc.copy_network(tensor_list) 
            best_network = cc.copy_network(tensor_list) 
            loss_record[0]["loss"] = first_loss
            loss_record.append({'iter':stage,
                'network':best_network,
                'num_params':cc.num_params(best_network),
                'loss':best_loss,
                'train_loss_hist':hist[0][:best_epoch]})

        else:
            m_print(f"\n\n**** Discrete optimization - iteration {stage} ****\n\n\n")
            # Create new TN random ranks
            ranks = torch.randint(low=1, high=max_rank+1, size=(n_edges,1)).view(-1,).tolist()
            rank_list = _make_ranks(ranks)
            currentNetwork = _limit_random_tn(input_dims, rank=rank_list, max_params=max_params)
            currentNetwork = cc.make_trainable(currentNetwork)
            if stop_on_plateau:
                detect_plateau._reset()
            currentNetwork, first_loss, current_loss, best_epoch, hist = training(currentNetwork, initial_epochs, train_data,
                    loss_fun, val_data=val_data, epochs=epochs, 
                    other_args=other_args)
            train_lost_hist = hist[0][:best_epoch]
            loss_record.append({'iter':stage,
                'network':currentNetwork,
                'num_params':cc.num_params(currentNetwork),
                'loss':current_loss,
                'train_loss_hist':train_lost_hist})
            if best_loss > current_loss:
                best_network = currentNetwork
                best_loss = current_loss
                
            print('\nbest TN:')
            cc.print_ranks(best_network)
            print('number of params:', cc.num_params(best_network))
            print([(r['iter'],r['num_params'],float(r['loss']),float(r['train_loss_hist'][0]),float(r['train_loss_hist'][-1])) for r in loss_record])


    return best_network, best_loss, loss_record

def randomsearch_decomposition(goal_tn):
    loss_fun = cc.tensor_recovery_loss
    input_dims = [t.shape[i] for i,t in enumerate(goal_tn)]
    base_tn = cc.random_tn(input_dims, rank=1)

    # Initialize the first tensor network close to zero
    for i in range(len(base_tn)):
        base_tn[i] /= 10
    base_tn = cc.make_trainable(base_tn)

    trained_tn, best_loss, loss_record = randomsearch_optim(base_tn, 
                                                      goal_tn, loss_fun, 
                                                      other_args={'cprint':True, 'epochs':10000, 'max_iter':20, 'max_rank':10, 
                                                                  'lr':0.01, 'optim':'RMSprop', 'search_epochs':80, 
                                                                  'cvg_threshold':1e-10, 
                                                                  'stop_on_plateau':{'mode':'min', 'patience':100, 'threshold':1e-7}, 
                                                                  'dyn_print':True,'initial_epochs':10})
    return loss_record

def randomsearch_regression(train_data, val_data):
    raise NotImplementedError