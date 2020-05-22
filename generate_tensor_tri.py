#!/usr/bin/env python
import tensornetwork as tn
import numpy as np
import torch

import core_code as cc

np.random.seed(1)
torch.manual_seed(1)

def generate_tensor_tri(input_dims, tri_ranks):
    """
    Generate random tensor network with tiangle structure
    
    Args:
        input_dims: List of input dimensions for each core in the network
        tri_ranks:  List of ranks
        
    Returns:
        tri_cores:  List of randomly initialized tensor cores for triangle TN
    """
    assert len(input_dims) == len(tri_ranks)
    n_cores = len(input_dims)
    ranks = []
    for i in range(n_cores-1):
            rank_i = np.ones((n_cores-1-i), dtype=np.int32)
            rank_i[0] = tri_ranks[i]
            ranks.append(rank_i.tolist())   
    ranks[-1][-1] = 1
    ranks[1][-1] = tri_ranks[-2]
    ranks[2][-1] = tri_ranks[-1]
    tri_cores = cc.random_tn(input_dims=input_dims, rank=ranks)
    
    return tri_cores 

if __name__ == '__main__':
    input_dims = [7,7,7,7,7]
    tri_ranks = [5,2,5,2,2]
    tri_cores = generate_tensor_tri(input_dims, tri_ranks)
    torch.save(tri_cores, 'tri_cores_5.pt')
