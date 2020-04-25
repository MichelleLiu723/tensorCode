import sys

import torch
import tensornetwork as tn

sys.path.append("..")
print(sys.path)
import core_code as cc

### EXAMPLE USAGE OF THE REVISED CODE ###

# The dimensions of each of the inputs to our tensor network (TN)
input_dims = [2, 4, 5, 6]

# Initialize a random rank-1 TN
example_tn = cc.random_tn(input_dims)

print("Rank-1 TN has ranks")
cc.print_ranks(example_tn)
print("...and input dimensions")
print(cc.get_indims(example_tn))


# Random TNs with higher ranks can also be defined
base_tn = cc.random_tn(input_dims, rank=3)

print("Rank-3 TN has ranks")
cc.print_ranks(base_tn)

# Individual ranks of TN edges can be set with upper-triangular format
rank_list = [[1,1,2], [3,5], [8]]
weird_tn = cc.random_tn(input_dims, rank=rank_list)

print("Weird TN has ranks")
cc.print_ranks(weird_tn)
print(f"It is defined by {cc.num_params(weird_tn)} real-valued parameters")


# To train, the tensor network cores must first be made trainable
base_tn = cc.make_trainable(base_tn)

# Let's implement the continuous optimization for tensor recovery. This 
# requires a target tensor, which we minimize L2 distance with
goal_tn = cc.random_tn(input_dims, rank=5)
goal_tn = cc.random_tn(input_dims, rank=rank_list)

# continuous_optim requires the following as input:
# (1) A tensor network model, base_tn
# (2) A target dataset, in this case just goal_tn
# (3) A loss function of the form loss_fun(base_tn, batch), where 
#     batch is a minibatch of training data (just goal_tn here)
# The choice of (2)+(3) fully determines the learning task, with other 
# problems actually using regular datasets as inputs

# For tensor recovery, use cc.loss_tensor_recovery for (3)
loss_fun = cc.loss_tensor_recovery
trained_tn, init_loss, final_loss = cc.continuous_optim(base_tn, goal_tn, 
                                                        loss_fun)
print(f"Train loss went from {init_loss:.3f} to {final_loss:.3f} in 10 epochs")

# Note that trained_tn gives the model after training, which will be 
# needed for discrete optimization algorithm

# 10 epochs is the default, but you can change this
cc.continuous_optim(base_tn, goal_tn, loss_fun, epochs=2)

# Feeding a dictionary as other_args arg of continuous_optim lets you 
# control the learning rate and optimizer (chosen from torch.optim)
my_args = {'optim': 'Adam',   # Default: 'SGD'
              'lr': 1e-2}     # Default: 1e-3
cc.continuous_optim(base_tn, goal_tn, loss_fun, other_args=my_args)

# You can also run optimization silently with the print argument
my_args['print'] = False      # Default: True
_, init_loss, final_loss = cc.continuous_optim(base_tn, goal_tn, loss_fun, 
                                               other_args=my_args)
print(f"Train loss went from {init_loss:.3f} to {final_loss:.3f} in 10 epochs")

# For tensor recovery, there is only one item in our dataset (goal_tn), 
# leading to only one gradient step per epoch. Using the `reps` argument 
# can reduce printing by going through training data many times per epoch
my_args = {'reps': 100}
# cc.continuous_optim(base_tn, goal_tn, loss_fun, other_args=my_args)


# For regression, (2) is a dataset of random inputs to a goal TN and noisy 
# scalar outputs. These need to be generated first
num_train = 10000
train_data = cc.generate_data(goal_tn, num_train)



# Mini-experiment to determine usefulness of different optimizers

# candidate_optims = ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 
#                     'Adamax', 'RMSprop', 'Rprop', 'SGD']
# percent_dec = {}
# my_args = {'print': False}
# print("Testing optimizers...")
# for optim in candidate_optims:
#     print("  " + optim)
#     my_args['optim'] = optim
#     my_args['lr']    = 1e-3
#     _, loss_i, loss_f = cc.continuous_optim(base_tn, goal_tn, loss_fun, 
#                                             epochs=100, other_args=my_args)
#     percent_dec[optim] = 100 * (loss_i-loss_f) / (loss_i)

# print("The Rankings in Loss Decrease are")
# percent_dec = sorted(percent_dec.items(), key=lambda p_dec: p_dec[1])
# for optim, p_dec in percent_dec:
#     print(f"  {optim+':':<9} {p_dec:.2f}% decrease")

