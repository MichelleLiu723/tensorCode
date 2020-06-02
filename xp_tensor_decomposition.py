#!/usr/bin/env python
import random
from functools import partial, reduce

import torch
import numpy as np
import tensornetwork as tn

import core_code as cc
from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os import path
import sys
import pickle

from utils import tic,toc

np.random.seed(0)
torch.manual_seed(0)


if len(sys.argv) < 2:
  print(f"usage: python {sys.argv[0]} [target_pytorch_file]\n \t target_pytorch_file: target network (list of tensors) saved in pytorch format")
target_file = sys.argv[1]

result_file = "results-" + path.splitext(path.basename(target_file))[0] + ".pickle"
if path.exists(result_file):
  print(f"output file already exists ({result_file})")
  sys.exit(-1)



goal_tn = torch.load(target_file)
target_full = cc.wire_network(goal_tn,give_dense=True).numpy()
input_dims = target_full.shape
target_TN_params=cc.num_params(goal_tn)
target_full_parms=np.prod(input_dims)

print('target tensor network number of params: ', target_TN_params)
print('number of params for full target tensor:', target_full_parms)
print('target tensor norm:', cc.l2_norm(goal_tn))
print('target tensor ranks:')
cc.print_ranks(goal_tn)

results = {"_xp-infos_":{
  'targt_TN_params':target_TN_params,
  'target_full_parms':target_full_parms,
  'target_network':goal_tn
}}


from tensor_decomposition_models import incremental_tensor_decomposition
from randomsearch import randomsearch_decomposition
from randomwalk import greedy_decomposition
from greedy import greedy_decomposition
for decomp in "randomsearch randomwalk greedy CP TT Tucker".split():
  print("*"*80)
  print(f"\n\n\nrunning {decomp} decomposition...\n")
  tic()
  if decomp == "randomsearch":
    results[decomp] = randomsearch_decomposition(goal_tn)
  elif decomp == "randomwalk":
    results[decomp] = randomwalk_decomposition(goal_tn)
  elif decomp == "greedy":
    results[decomp] = greedy_decomposition(goal_tn)
  else:
    results[decomp] = incremental_tensor_decomposition(target_full,decomp,verbose=True,max_num_params=3000, 
      rank_increment_factor=1.5 if decomp=='CP' else 1)
  results['_xp-infos_'][decomp + "-runtime"] = toc()
with open(result_file, "wb") as f:
  pickle.dump(results,f)

