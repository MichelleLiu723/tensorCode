import torch
import tensornetwork as tn
import core_code as cc

### EXAMPLE USAGE OF THE REVISED CODE ###

# The dimensions of each of the inputs to our tensor network (TN)
input_dims = [2, 3, 4, 5]

# Create a random rank-1 TN and make it trainable
base_tn = cc.random_tn(input_dims)
base_tn = cc.make_trainable(base_tn)
print("Rank-1 TN has ranks:")
cc.print_ranks(base_tn)


# Random TNs with higher ranks can also be defined. 
unused_tn = cc.random_tn(input_dims, rank=2)

print("Rank-2 TN has ranks:")
cc.print_ranks(unused_tn)

# You can specify the ranks of all TN edges individually, in this format
rank_list = [[1,2,3], [4,5], [6]]
unused_tn = cc.random_tn(input_dims, rank=rank_list)

print("Weird TN has ranks:")
cc.print_ranks(unused_tn)


# Let's implement the continuous optimization for tensor recovery
goal_tn = cc.random_tn(input_dims)

# continuous_optim requires a loss function as input, which specifies the
# type of problem we're solving. Here, use `loss_tensor_recovery`
recovery_loss = cc.loss_tensor_recovery
trained_tn, init_loss, final_loss = cc.continuous_optim(base_tn, goal_tn, 
                                                        recovery_loss)
print(f"Train loss went from {init_loss:.3f} to {final_loss:.3f} after 10 epochs")

# 10 epochs is the default, but you can change this
# cc.continuous_optim(base_tn, goal_tn, recovery_loss, epochs=2)

# There is only one item in our dataset (goal_tn), producing only one
# gradient step per epoch (slow!). Using the `reps` argument can speed this up
other_args = {'reps': 100}
cc.continuous_optim(base_tn, goal_tn, recovery_loss, other_args=other_args)