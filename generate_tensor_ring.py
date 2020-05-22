#!/usr/bin/env python
import tensornetwork as tn
import numpy as np
import torch

import core_code as cc

np.random.seed(1)
torch.manual_seed(1)

def generate_tensor_ring(input_dims, tr_ranks):
    """
    Generate random tensor ring
    
    Args:
        input_dims: List of input dimensions for each core in the network
        tr_ranks:   List of TR ranks
        
    Returns:
        tr_cores:   List of randomly initialized tensor ring cores
    """
    assert len(input_dims) == len(tr_ranks)
    n_cores = len(input_dims)
    ranks = []
    for i in range(n_cores-1):
            rank_i = np.ones((n_cores-1-i), dtype=np.int32)
            rank_i[0] = tr_ranks[i]
            ranks.append(rank_i.tolist())
    ranks[0][-1] = tr_ranks[-1]
    tr_cores = cc.random_tn(input_dims=input_dims, rank=ranks)
    
    return tr_cores 

if __name__ == '__main__':
    input_dims = [7,7,7,7,7]
    tr_ranks = [2,3,4,5,4]
    tr_cores = generate_tensor_ring(input_dims, tr_ranks)
    torch.save(tr_cores, 'tr_cores_5.pt')