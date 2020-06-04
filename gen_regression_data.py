#!/usr/bin/env python
import numpy as np
import torch

import core_code as cc

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    target_file = 'tt_cores_5.pt'  # 'tr_cores_5.pt' 'tri_cores_5.pt'
    goal_tn = torch.load(target_file)
    num_train = 20000
    num_val = 2000
    train_data = cc.generate_regression_data(goal_tn, num_train, noise=1e-6)
    val_data   = cc.generate_regression_data(goal_tn, num_val,   noise=1e-6)
    torch.save({'train_data':train_data, 'val_data':val_data}, 'reg_data-'+target_file)