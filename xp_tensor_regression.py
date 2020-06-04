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


if len(sys.argv) < 3:
  print(f"usage: python {sys.argv[0]} [target_pytorch_file]\n \t target_pytorch_file: target regression data saved in pytorch format\n \t num_train: number  of training samples")
target_file = sys.argv[1]
num_train = int(sys.argv[2])

result_file = "results-" + path.splitext(path.basename(target_file))[0] +'-'+ str(num_train) +".pickle"

if path.exists(result_file):
  print(f"output file already exists ({result_file})")
  sys.exit(-1)

def sample_reg_data(reg_data, num_train):
    num_val = num_train//10
    train_data = reg_data['train_data']
    val_data = reg_data['val_data'][:num_val]
    
    train_weights = [t[:num_train] for t in reg_data['train_data'][0]]
    train_labels = reg_data['train_data'][1][:num_train]
    train_data = (train_weights, train_labels)
    
    val_weights = [t[:num_val] for t in reg_data['val_data'][0]]
    val_labels = reg_data['val_data'][1][:num_val]
    val_data = (val_weights, val_labels)
    return train_data, val_data


reg_data = torch.load(target_file)
train_data, val_data = sample_reg_data(reg_data, num_train)
goal_tn = torch.load(target_file.split('-')[1])
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


from randomsearch import randomsearch_regression
from randomwalk import randomwalk_regression
from greedy import greedy_regression
for regress in "randomwalk greedy".split():
  print("*"*80)
  print(f"\n\n\nrunning {regress} regression...\n")
  tic()
  if regress == "randomsearch":
    results[regress] = randomsearch_regression(train_data, val_data)
  elif regress == "randomwalk":
    results[regress] = randomwalk_regression(train_data, val_data)
  elif regress == "greedy":
    results[regress] = greedy_regression(train_data, val_data)
  results['_xp-infos_'][regress + "-runtime"] = toc()
with open(result_file, "wb") as f:
  pickle.dump(results,f)

