import torch
import tensornetwork as tn
from core_code import *

input_dims = [2, 3, 4, 5, 6]
base_tn = random_tn(input_dims)
goal_tn = random_tn(input_dims)

continuous_optim(base_tn, goal_tn, loss_tensor_recovery)