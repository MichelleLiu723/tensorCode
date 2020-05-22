#!/usr/bin/env python
import tensornetwork as tn
import numpy as np
import torch

import core_code as cc

np.random.seed(1)
torch.manual_seed(1)

def generate_tensor_train(input_dims, tt_ranks):
    """
    Generate random tensor train
    
    Args:
        input_dims: List of input dimensions for each core in the network
        tt_ranks:   List of TT ranks
        
    Returns:
        tt_cores:   List of randomly initialized tensor train cores
    """
    assert len(input_dims) == len(tt_ranks)+1
    n_cores = len(input_dims)
    ranks = []
    for i in range(n_cores-1):
            rank_i = np.ones((n_cores-1-i), dtype=np.int32)
            rank_i[0] = tt_ranks[i]
            ranks.append(rank_i.tolist())
    tt_cores = cc.random_tn(input_dims=input_dims, rank=ranks)
    
    return tt_cores 

if __name__ == '__main__':
    input_dims = [7,7,7,7,7]
    tt_ranks = [2,3,6,5]
    tt_cores = generate_tensor_train(input_dims, tt_ranks)
    torch.save(tt_cores, 'tt_cores_5.pt')
