#!/usr/bin/env python 
import random
from inspect import signature
from functools import partial, reduce

import torch
import numpy as np
import tensornetwork as tn
import torch.optim as optim


"""
### NOTES ON TENSOR NETWORK FORMAT ###

A tensor network is specified as a list of tensors, each of whose shape is
formatted in a specific way. In particular, for a network with n tensor
cores, the shape will be:   

tensor_i.shape = (r_1, r_2, ..., r_i, ..., r_n),

where r_j gives the tensor network rank (TN-rank) connecting cores i and j
when j != i, while r_i := d_i gives the dimension of the input to core i. 
This implies that tensor_i.shape[j] == tensor_j.shape[i], and stacking the 
shapes of all the tensors in order gives the adjacency matrix of the 
network as follows:

     [[d_1, r_{1,2}, ..., r_{1,n}],
      [r_{1,2}, d_2, ..., r_{2,n}],
                     ...
      [r_{1,n}, r_{2,n}, ..., d_n]]

The diagonal entries here are precisely the input dimensions for each of 
the core tensors, while the symmetric nature of the matrix is necessary 
for different cores to contract with each other.

Besides the above matrix format, the ranks are also sometimes specified 
in the following upper-triangular format:

     [[r_{1,2}, r_{1,3}, ..., r_{1,n}], [r_{2,3}, ..., r_{2,n}], ...
      ..., [r_{n-2,n-1}, r_{n-2,n}], [r_{n-1,n}]]
"""

# Set global defaults
tn.set_default_backend("pytorch")
torch.set_default_tensor_type(torch.DoubleTensor)

def continuous_optim(tensor_list, train_data, loss_fun, epochs=10, 
                     val_data=None, other_args=dict()):
    """
    Train a tensor network using gradient descent on input dataset

    Args:
        tensor_list: List of tensors encoding the network being trained
        train_data:  The data used to train the network
        loss_fun:    Scalar-valued loss function of the type 
                        tens_list, data -> scalar_loss
                     (This depends on the task being learned)
        epochs:      Number of epochs to train for. When val_data is given,
                     setting epochs=None implements early stopping
        val_data:    The data used for validation
        other_args:  Dictionary of other arguments for the optimization, 
                     with some options below (feel free to add more)

                        optim: Choice of Pytorch optimizer (default='SGD')
                        lr:    Learning rate for optimizer (default=1e-3)
                        bsize: Minibatch size for training (default=100)
                        reps:  Number of times to repeat 
                               training data per epoch     (default=1)
                        print: Whether to print info       (default=True)
                        dyn_print: use dynamic printing    (default=False)
                        hist:  Whether to return losses
                               from train and val sets     (default=False)
                        momentum: Momentum value for 
                                  continuous optimization  (default=0)
                        cvg_threshold: threshold to test convergence of 
                            optimization (optimization is stopped if 
                            |(prev_loss - cur_loss)/prev_loss| <  cvg_threshold
                            If None, convergence is not checked. If epochs is
                            set as well, then optimziation is stopped either when
                            convergence criteria is met or when epochs is reached
                                                            (default:None)
                        lr_scheduler: a function taking an optimizer as input
                        and returning a learning rate scheduler for this optimizer
                                                            (default:None)
                        save_optimizer_state: if True, other_args should have an empty
                            dict for the key optimizer_state. This dict will contain 
                              {optimizer_state: optimizer state_dict,
                              lr_scheduler_state: scheduler state_dict (if any)}
                            after the function returns.     (default:False)
                        load_optimzer_state: a dictionnary that will be used to 
                            initialize the optimizer (and scheduler if any) from a
                            previously saved optimizer state.
                                                            (default: None)
                        grad_masking_function: a function taking the list of tensor
                            parameters between the backward pass and the optimizer step
                            (can be used to e.g. zero out parts of the gradient)
                                                            (default: None)
                        stop_condition: a function taking the training and validation loss
                            as input after each epoch and returning True if optimization 
                            should be stopped               (default: None)

    
    Returns:
        better_list: List of tensors with same shape as tensor_list, but
                     having been optimized using the appropriate optimizer.
                     When validation data is given, the model with the 
                     lowest validation loss is output, otherwise the model
                     with lowest training loss
        first_loss:  Initial loss of the model on the validation set, 
                     before any training. If no val set is provided, the
                     first training loss is instead returned
        best_loss:   The value of the validation/training loss for the
                     model output as better_list
        best_epoch:  epoch at which best_model was found
        loss_record: If hist=True in other_args, history of all validation
                     and training losses is returned as a tuple of Pytorch
                     vectors (train_loss, val_loss), with each vector
                     having length equal to number of epochs of training.
                     When no validation loss is provided, the second item
                     (val_loss) is an empty tensor.
    """
    # Check input and initialize local record variables
    early_stop = epochs is None
    has_val = val_data is not None
    optim = other_args['optim'] if 'optim' in other_args else 'SGD'
    lr    = other_args['lr']    if 'lr'    in other_args else 1e-3
    bsize = other_args['bsize'] if 'bsize' in other_args else 100
    reps  = other_args['reps']  if 'reps'  in other_args else 1
    prnt  = other_args['print'] if 'print' in other_args else True
    hist  = other_args['hist']  if 'hist'  in other_args else False
    dyn_print  = other_args['dyn_print']  if 'dyn_print'  in other_args else False
    lr_scheduler  = other_args['lr_scheduler']  if 'lr_scheduler'  in other_args else None
    cvg_threshold  = other_args['cvg_threshold']  if 'cvg_threshold'  in other_args else None
    save_optimizer_state  = other_args['save_optimizer_state']  if 'save_optimizer_state'  in other_args else None
    load_optimizer_state  = other_args['load_optimizer_state']  if 'load_optimizer_state'  in other_args else None
    grad_masking_function  = other_args['grad_masking_function']  if 'grad_masking_function'  in other_args else None
    momentum = other_args['momentum'] if 'momentum' in other_args else 0


    stop_condition  = other_args['stop_condition']  if 'stop_condition'  in other_args else None

    if save_optimizer_state and (not 'optimizer_state' in other_args):
        raise ValueError("an empty dictionnary should be passed as the optimizer_state argument to store the"
            " optimizer state.")
    if early_stop and not has_val:
         raise ValueError("Early stopping (epochs=None) requires val_data "
                          "to be input")
    loss_rec, first_loss, best_loss, best_network, best_epoch = [], None, np.infty, tensor_list, 0
    if hist: loss_record = ([], [])    # (train_record, val_record)

    # Function to maybe print, conditioned on `prnt`
    m_print = lambda s: print(s, end='\r' if dyn_print else '\n') if prnt else None

    # Function to record loss information and return whether to stop
    def record_loss(new_loss, new_network, epoch_num):
        # Load record variables from outer scope
        nonlocal loss_rec, first_loss, best_loss, best_network, best_epoch

        # Check for first and best loss
        if best_loss is None or new_loss < best_loss:
            best_loss, best_network, best_epoch = new_loss, new_network, epoch_num
        if first_loss is None:
            first_loss = new_loss

        # Update loss record and check for early stopping. If you want to
        # change early stopping criteria, this is the place to do it.
        window = 2    # Number of epochs kept for checking early stopping
        warmup = 1    # Number of epochs before early stopping is checked
        if len(loss_rec) < window:
            stop, loss_rec = False, loss_rec + [new_loss]
        else:
            # stop = new_loss > sum(loss_rec)/len(loss_rec)
            stop = (new_loss > max(loss_rec)) and (epoch_num >= warmup)
            loss_rec = loss_rec[1:] + [new_loss]

        return stop

    # Another loss logging function, but for recording *all* loss history
    @torch.no_grad()
    def loss_history(new_loss, is_val):
        if not hist: return
        nonlocal loss_record
        loss_record[int(is_val)].append(new_loss)

    # Function to run TN on validation data
    @torch.no_grad()
    def run_val(t_list):
        val_loss = []

        # Note that `batchify` uses different logic for different types
        # of input, so update batchify when you work on tensor completion
        for batch in batchify(val_data):
            val_loss.append(loss_fun(t_list, batch))
        if has_val:
            val_loss = torch.mean(torch.tensor(val_loss))

        return val_loss

    # Copy tensor_list so the original is unchanged
    tensor_list = copy_network(tensor_list)

    # Record the initial validation loss (if we validation dataset)
    if has_val: record_loss(run_val(tensor_list), tensor_list, 0)


    # Initialize optimizer, using only the keyword args in the 
    optim = getattr(torch.optim, optim)
    opt_args = signature(optim).parameters.keys()
    kwargs = {'lr':lr, 'momentum':momentum}    # <- Add new options here
    kwargs = {k: v for (k, v) in kwargs.items() if k in opt_args}
    optim = optim(tensor_list, **kwargs)    # Initialize the optimizer
    if lr_scheduler: # instantiate learning rate scheduler
        scheduler = lr_scheduler(optim)

    if load_optimizer_state:
        optim.load_state_dict(other_args["load_optimizer_state"]["optimizer_state"])
        if lr_scheduler:
            scheduler.load_state_dict(other_args["load_optimizer_state"]["lr_scheduler_state"])

    # Loop over validation and training for given number of epochs
    ep = 1
    prev_loss = np.infty



    while epochs is None or ep <= epochs:

        # Train network on all the training data
        #from copy import deepcopy
        prev_tensor_list = copy_network(tensor_list)
        #prev_tensor_list = tensor_list
        train_loss, num_train = 0., 0
        for batch in batchify(train_data, batch_size=bsize, reps=reps):
            loss = loss_fun(tensor_list, batch)
            optim.zero_grad()
            loss.backward()
            if grad_masking_function:
                grad_masking_function(tensor_list)
            optim.step()

            with torch.no_grad():
                num_train += 1
                train_loss += loss

        train_loss /= num_train

        if lr_scheduler:
            scheduler.step(train_loss)

        loss_history(train_loss, is_val=False)
        
        val_loss = run_val(tensor_list) if has_val else None

        val_loss_str = f"Val. loss:  {val_loss.data:.10f}" if has_val else ""
        m_print(f"EPOCH {ep} {'('+str(reps)+' reps)' if reps > 1 else ''}\t\t{val_loss_str}\t\t Train loss: {train_loss.data:.10f}\t\t Convergence: {np.abs(train_loss-prev_loss)/prev_loss:.10f}")

        # Get validation loss if we have it, otherwise record training loss
        if has_val:
            # Get and record validation loss, check early stopping condition
            loss_history(val_loss, is_val=True)
            if record_loss(val_loss, copy_network(tensor_list) if has_val else prev_tensor_list, ep) and early_stop:
                print(f"\nEarly stopping condition reached")
                break
        else:
            record_loss(train_loss, copy_network(tensor_list) if has_val else prev_tensor_list, ep)

        if cvg_threshold and np.abs(train_loss-prev_loss)/prev_loss < cvg_threshold:
            print(f"\nConvergence criteria reached")
            break
        if stop_condition and stop_condition(train_loss=train_loss,val_loss=val_loss):
            print(f"\nStopping condition reached")
            break

        prev_loss = train_loss

    

        ep += 1
    m_print("")

    # Save the optimizer state if needed
    if save_optimizer_state:
        other_args["optimizer_state"]["optimizer_state"] = optim.state_dict()
        if lr_scheduler:
            other_args["optimizer_state"]["lr_scheduler_state"] = scheduler.state_dict()

    if hist:
        loss_record = tuple(torch.tensor(fr) for fr in loss_record)
        return best_network, first_loss, best_loss, best_epoch, loss_record
    else:
        return best_network, first_loss, best_loss

def contract_network(nodes, contractor='auto', edge_order=None):
    """
    Contract a tensor network that has already been 'wired' together

    Args:
        nodes:      One or more nodes in the network of interest. All 
                    nodes connected to this one will get contracted
        contractor: Name of the TensorNetwork contractor used to contract
                    the network. Options include 'greedy', 'optimal', 
                    'bucket', 'branch', and 'auto' (default)
        edge_order: When expanding to a dense tensor, giving a list of the
                    dangling edges of the tensor is required to set the
                    order of the indices of the (large) output tensor

    Returns:
        output:     Pytorch tensor containing the contracted network, 
                    which here will always be a scalar or batch vector
    """
    contractor = getattr(tn.contractors, contractor)
    output = contractor(tn.reachable(nodes), output_edge_order=edge_order)
    return output.tensor

def evaluate_input(tensor_rep, input_list):
    """
    Contract input vectors with large tensor to get scalar output

    Args:
        tensor_rep:  The tensor network whose input indices are being 
                     contracted. This currently only supports TN input 
                     (i.e. list of properly formatted tensor cores), but
                     please ask if you want support for inputs that are
                     specified as one big dense tensor
        input_list:  Batch of inputs to feed to the cores in our tensor.
                     This can be either a list of matrices with shapes 
                     (batch_dim, input_dim_i) or a single Pytorch tensor 
                     with shape (num_cores, batch_dim, input_dim)

    Returns:
        closed_list: Scalar or batch tensor giving output of contraction
                     between tensor network and input data
    """
    num_cores = len(tensor_rep)
    assert len(input_list) == num_cores
    assert isinstance(tensor_rep, (tuple, list, torch.Tensor))
    assert len(set(len(inp.shape) for inp in input_list)) == 1

    # Get batch information about our input
    input_shape = input_list[0].shape
    has_batch = len(input_shape) == 2
    assert len(input_shape) in (1, 2)
    if has_batch: 
        batch_dim = input_shape[0]
        assert all(i.shape[0] == batch_dim for i in input_list)

        # Generate batch node for dealing with batch dims
        batch_edges = batch_node(num_cores, batch_dim)
        assert len(batch_edges) == num_cores + 1

    # Convert all tensor cores to list of wired nodes
    node_list = wire_network(tensor_rep)

    # Go through and contract all inputs with corresponding cores
    for i, node, inp in zip(range(num_cores), node_list, input_list):
        inp_node = tn.Node(inp)
        node[i] ^ inp_node[int(has_batch)]

        # Explicitly contract batch indices together if we need that
        if has_batch:
            inp_node[0] ^ batch_edges[i]

    return contract_network(node_list)

def tn_inner_prod(tensor_list1, tensor_list2):
    """
    Get inner product of two tensor networks with identical input dims

    Args:
        tensor_list1: List of (properly formatted) tensors that encodes
                      the first (open) tensor network
        tensor_list2: List of (properly formatted) tensors that encodes
                      the second (open) tensor network

    Returns:
        inner_prod:   Scalar giving inner product between input networks
    """
    num_cores = len(tensor_list1)
    assert len(tensor_list1) == len(tensor_list2)
    assert all(tensor_list1[i].shape[i] == tensor_list2[i].shape[i] 
                for i in range(num_cores))

    # Wire up each of the networks
    network1 = wire_network(tensor_list1)
    network2 = wire_network(tensor_list2)

    # Contract all input indices together
    for i in range(num_cores):
        network1[i][i] ^ network2[i][i]

    return contract_network(network1 + network2)

def l2_norm(tensor_list):
    """Compute the Frobenius norm of tensor network"""
    return torch.sqrt(tn_inner_prod(tensor_list, tensor_list))

def l2_distance(tensor_list1, tensor_list2):
    """
    Compute L2 distance between two tensor networks with same input dims
    """
    norm1, norm2 = l2_norm(tensor_list1), l2_norm(tensor_list2)
    inner_prod = tn_inner_prod(tensor_list1, tensor_list2)

    return torch.sqrt(norm1**2 + norm2**2 - 2*inner_prod)

def wire_network(tensor_list, give_dense=False):
    """
    Convert list of tensor cores into fully wired network of TN Nodes

    If give_dense=True, the wired network is contracted together and a 
    single (large) tensor is returned
    """
    num_cores = len(tensor_list)
    assert valid_formatting(tensor_list)

    # Wire together all internal edges connecting cores
    node_list = [tn.Node(core) for core in tensor_list]
    for i in range(num_cores):
        for j in range(i+1, num_cores):
            node_list[i][j] ^ node_list[j][i]

    if give_dense:
        edge_order = [node[i] for i, node in enumerate(node_list)]
        return contract_network(node_list, edge_order=edge_order)
    else:
        return node_list

def expand_network(tensor_list):
    """
    Contract all cores of a tensor network together, yielding dense tensor
    """
    return wire_network(tensor_list, give_dense=True)

def random_tn(input_dims=None, rank=1):
    """
    Initialize a tensor network with random (normally distributed) cores

    Args:
        input_dims:  List of input dimensions for each core in the network
        rank:        Scalar or list of rank connecting different cores.
                     For scalar inputs, all ranks will be initialized at
                     the specified number, whereas more fine-grained ranks
                     are specified via a square matrix, or in the 
                     following triangular format:
                     [[r_{1,2}, r_{1,3}, ..., r_{1,n}], [r_{2,3}, ..., r_{2,n}],
                      ..., [r_{n-2,n-1}, r_{n-2,n}], [r_{n-1,n}]]

    Returns:
        tensor_list: List of randomly initialized and properly formatted
                     tensors encoding our tensor network
    """
    # Convert rank object to list of shapes
    if hasattr(rank, 'shape'):
        # Matrix format
        shape_mat = rank.shape
        assert len(shape_mat) == 2
        assert shape_mat[0] == shape_mat[1]
        shape_list = [tuple(int(r) for r in row) for row in shape_mat]
    elif hasattr(rank, '__len__'):
        if len(set(len(row) for row in rank)) == 1:
            # Matrix-type format
            shape_list = [tuple(int(r) for r in row) for row in rank]
        else:
            # Upper-triangular format 
            shape_list = unpack_ranks(input_dims, rank) 
    else:
        # Scalar format
        assert hasattr(input_dims, '__len__')
        r, n_c = rank, len(input_dims)
        shape_list = [(r,)*i + (d,) + (r,)*(n_c-1-i) 
                      for i, d in enumerate(input_dims)]
    num_cores = len(shape_list)

    # Check that diagonals match input_dims
    if input_dims is not None:
        assert len(input_dims) == num_cores
        assert all(shape_list[i][i] == d for i, d in enumerate(input_dims))

    # Heuristic function used to set stdev of normally-distributed tensor
    # elements, feel free to replace as desired
    def stdev_fun(shape):
        # Square root of num_core'th root of number of elements in shape
        num_el = torch.prod(torch.tensor(shape)).to(dtype=torch.float)
        return torch.exp(torch.log(num_el) / (2 * num_cores))

    # Use shapes to instantiate random core tensors
    # The variance of the normal distributions is chosen to make the tensor norm 1 in expectation.
    tensor_list = []
    for i,shape in enumerate(shape_list):
        std = 1/np.power(torch.prod(torch.tensor(shape)) * shape[i],0.25).to(dtype=torch.float)
        tensor_list.append(std * torch.randn(shape))

    return tensor_list

make_trainable = lambda tensor_list: [t.requires_grad_() 
                                      for t in tensor_list]
make_trainable.__doc__ = "Returns trainable version of tensor list"

def copy_network(tensor_list): 
    """Returns detached copy of tensor list"""
    my_copy = [t.clone().detach() for t in tensor_list]
    my_copy = [ct.requires_grad_() if t.requires_grad else ct 
               for t, ct in zip(tensor_list, my_copy)]
    return my_copy

def generate_regression_data(target_tensor, num_data, noise=1e-5):
    """
    Use a target tensor network to get pair of batch (input, output) data

    Args:
        target_tensor: List of tensors encoding a target tensor network,
                       which is used to generate the data
        num_data:     The number of inputs and outputs to generate
        noise:         Stdev of Guassian noise to add to real output value

    Return:
        rand_ins:      List of matrices, with i'th entry having shape
                       (num_data, input_dim_i), with input_dim_i being 
                       the i'th visible dimension of target_tensor
        rand_out:      Vector of length num_data holding noisy outputs
    """
    num_cores = len(target_tensor)
    in_dims = torch.tensor(get_indims(target_tensor))
    rand_ins = [torch.randn((num_data, d)) / torch.sqrt(d.double()) 
                    for d in in_dims]
        
    # Convert into tensors when possible
    if len(set(in_dims)) == 1:
        rand_ins = torch.tensor(rand_ins)

    # Produce outputs in small batches
    eval_fun = partial(evaluate_input, target_tensor)
    num, mini_size, rand_out = 0, 50, []
    while num < num_data:
        this_in = [r_in[num:num+mini_size] for r_in in rand_ins]
        rand_out.append(evaluate_input(target_tensor, this_in))
        num += mini_size
    rand_out = torch.cat(rand_out)

    # Add noise to output data
    rand_out += noise * torch.randn_like(rand_out)

    return rand_ins, rand_out

def generate_completion_data(target_tensor, num_data, noise=1e-5):
    """
    Use a target tensor network to get dataset of tensor elements

    Args:
        target_tensor: List of tensors encoding a target tensor network,
                       which is used to generate the data
        num_data:      The number of elements and values to generate
        noise:         Stdev of Guassian noise to add to tensor elements

    Return:
        rand_elms:     Random integer-valued Pytorch matrix of shape 
                       (num_cores, num_data) containing random elements
                       that target_tensor is evaluated on
        rand_vals:     Vector of length num_data holding noisy values of 
                       target_tensor at elements in rand_elms
    """
    num_cores = len(target_tensor)
    in_dims = torch.tensor(get_indims(target_tensor))

    # Generate rand_elms, then convert into one-hot format
    rand_elms = torch.stack([torch.randint(d, (num_data,)) for d in in_dims])
    one_hot = torch.functional.F.one_hot
    rand_ins = [one_hot(vec, d).double()
                    for vec, d in zip(rand_elms, in_dims)]
    
    # Produce outputs in small batches
    eval_fun = partial(evaluate_input, target_tensor)
    num, mini_size, rand_vals = 0, 50, []
    while num < num_data:
        this_in = [r_in[num:num+mini_size] for r_in in rand_ins]
        rand_vals.append(evaluate_input(target_tensor, this_in))
        num += mini_size
    rand_vals = torch.cat(rand_vals)

    # Add noise to output data
    rand_vals += noise * torch.randn_like(rand_vals)

    return rand_elms, rand_vals

def unpack_ranks(in_dims, ranks):
    """Converts triangular `ranks` structure to list of tensor shapes"""
    num_cores = len(in_dims)
    assert [len(rl) for rl in ranks] == list(range(num_cores - 1, 0, -1))

    shape_list = []
    for i in range(num_cores):
        shape = [ranks[j][i-j-1] for j in range(i)]               # Lower triangular
        shape += [ranks[i][j-i-1] for j in range(i+1, num_cores)] # Upper triangular
        shape.insert(i, in_dims[i])                               # Diagonals
        shape_list.append(tuple(shape))

    return shape_list

def print_ranks(tensor_list):
    """Print out the ranks of edges in tensor network"""
    all_shapes = torch.tensor([t.shape for t in tensor_list])
    print(torch.triu(all_shapes).numpy())

def get_indims(tensor_list):
    """Return the input dimensions of nodes in tensor network"""
    all_shapes = torch.tensor([t.shape for t in tensor_list])
    return tuple(int(s[i]) for i, s in enumerate(all_shapes))

def valid_formatting(tensor_list):
    """Check if a tensor list is correctly formatted"""
    num_cores = len(tensor_list)
    try:
        shape_mat = torch.tensor([core.shape for core in tensor_list])
        assert torch.all(shape_mat == shape_mat.T)
        return True
    except:
        return False

def batch_node(num_inputs, batch_dim):
    """
    Return a network of small CopyNodes which emulates a large CopyNode

    This network is used for reproducing the standard batch functionality 
    available in Pytorch, and requires connecting the `num_inputs` edges
    returned by batch_node to the respective batch indices of our inputs.
    The sole remaining dangling edge will then give the batch index of 
    whatever contraction occurs later with the input.

    Args:
        num_inputs: The number of batch indices to contract together
        batch_dim:  The batch dimension we intend to reproduce

    Returns:
        edge_list:  List of edges of our composite CopyNode object
    """
    # For small numbers of edges, just use a single CopyNode
    num_edges = num_inputs + 1
    if num_edges < 4:
        node = tn.CopyNode(rank=num_edges, dimension=batch_dim)
        return node.get_all_edges()

    # Initialize list of free edges with output of trivial identity mats
    input_node = tn.Node(torch.eye(batch_dim))
    edge_list, dummy_list = zip(*[input_node.copy().get_all_edges() 
                                  for _ in range(num_edges)])

    # Contract dummy edges as a binary tree via third-order tensors
    dummy_len = len(dummy_list)
    while dummy_len > 4:
        odd = dummy_len % 2 == 1
        half_len = dummy_len // 2

        # Apply third order tensor to contract two dummy indices together
        temp_list = []
        for i in range(half_len):
            temp_node = tn.CopyNode(rank=3, dimension=batch_dim)
            temp_node[1] ^ dummy_list[2 * i]
            temp_node[2] ^ dummy_list[2 * i + 1]
            temp_list.append(temp_node[0])
        if odd:
            temp_list.append(dummy_list[-1])

        dummy_list = temp_list
        dummy_len = len(dummy_list)

    # Contract the 3 or less dummy indices together
    last_node = tn.CopyNode(rank=dummy_len, dimension=batch_dim)
    [last_node[i] ^ dummy_list[i] for i in range(dummy_len)]

    return edge_list

def increase_rank(slim_list, vertex1, vertex2, rank_inc=1, pad_noise=1e-6):
    """
    Increase the rank of one bond in a tensor network

    Args:
        slim_list: List of tensors encoding a tensor network
        vertex1:   Node number for one end of the edge being increased
        vertex2:   Node number for the other end of the edge being 
                   increased, which can't equal vertex1
        rank_inc:  Amount to increase the rank by (default 1)
        pad_noise: Increasing the rank involves embedding the original
                   TN in a larger parameter space, and adding a bit of 
                   noise (set by pad_noise) helps later in training

    Returns:
        fat_list:  List of tensors encoding the same network, but with the
                   rank of the edge connecting nodes vertex1 and vertex2 
                   increased by rank_inc
    """
    num_tensors = len(slim_list)
    assert 0 <= vertex1 < num_tensors
    assert 0 <= vertex2 < num_tensors
    assert rank_inc >= 0
    assert pad_noise >= 0

    # Function for increasing one index of one tensor
    def pad_tensor(tensor, ind):
        shape = list(tensor.shape)
        shape[ind] = rank_inc
        pad_mat = torch.randn(shape) * pad_noise
        return torch.cat([tensor, pad_mat], dim=ind)

    # Pad both of the tensors along the index of the other
    fat_list = slim_list
    fat_list[vertex1] = pad_tensor(fat_list[vertex1], vertex2)
    fat_list[vertex2] = pad_tensor(fat_list[vertex2], vertex1)

    return fat_list

def num_params(tensor_list):
    """
    Get number of parameters associated with a tensor network

    Args:
        tensor_list: List of tensors encoding a tensor network

    Return:
        param_count: The number of parameters in the tensor network
    """
    return sum(t.numel() for t in tensor_list)

def tensor_recovery_loss(tensor_list, target_tensor):
    """
    Compute the L2 distance between our tensor network and a target network

    Args:
        tensor_list:   List of tensors encoding a tensor network
        target_tensor: The tensor network we are trying to approximate

    Returns:
        loss:          The L2 distance

    """
    return l2_distance(tensor_list, target_tensor)

def regression_loss(tensor_list, dataset, p=2):
    """
    Compute the regression loss of TN when used as scalar-valued fun, 
    relative to dataset of inputs and outputs

    Args:
        tensor_list: List of tensors encoding a tensor network
        dataset:     Tuple of the form (input_list, targets), with targets 
                     a Pytorch vector containing target values, and 
                     input_list either a list of matrices with shapes 
                     (batch_dim, input_dim_i), or else a single Pytorch 
                     tensor with shape (num_cores, batch_dim, input_dim)
        p:           Sets which p-norm is used for the loss (default: 2)

    Returns:
        loss:        The sum of distance between targets and the output of
                     tensor_list when fed the contents of input_list
    """
    # Unpack dataset and evaluate inputs using tensor network
    input_list, targets = dataset
    num_ins = len(targets)
    outputs = evaluate_input(tensor_list, input_list)
    
    return torch.dist(targets, outputs, p=p)

def completion_loss(tensor_list, dataset, p=2, use_full_tensor=True):
    """
    Compute sum of distances between actual and target tensor elements, 
    for all elements with known values in the dataset

    Args:
        tensor_list: List of tensors encoding a tensor network
        dataset:     Tuple of the form (input_elms, targets), with targets 
                     a Pytorch vector of target values, and input_elms an 
                     integer Pytorch matrix of shape (num_cores, batch_dim)
        p:           Sets which p-norm is used for the loss (default: 2)
        use_full_tensor: if True the tensor_list is contracted to a full tensor
            to extract observed entries (does not scale to very large tensors!),
            if False, efficient contractions with one hots is done.

    Returns:
        loss:        The sum of distance between targets and the output of
                     tensor_list when fed the contents of input_list
    """
    # Convert input_elms to one-hot vectors, then call regression loss
    if use_full_tensor:
        full_tensor = wire_network(tensor_list,give_dense=True)
        idx, vals = dataset
        return torch.dist(full_tensor[idx.chunk(chunks=idx.shape[-1])],vals, p=p)
    else:
        input_elms, targets = dataset
        in_dims = get_indims(tensor_list)
        one_hot = torch.functional.F.one_hot
        input_list = [one_hot(vec, d).double()
                          for vec, d in zip(input_elms, in_dims)]
        return regression_loss(tensor_list, (input_list, targets), p=p)

@torch.no_grad()
def batchify(dataset, batch_size=100, reps=1):
    """
    Convert dataset into iterator over minibatches of data

    The nature of this iteration depends on the task at hand, so batchify
    should be modified when new tasks are proposed. Currently supported
    options are tensor recover and regression
    """
    # Figure out which task we're dealing with
    if dataset is None: return  # Trivial input data with no iteration

    elif isinstance(dataset, list) and valid_formatting(dataset):
        task = 'recovery'
    elif len(dataset) == 2 and isinstance(dataset[0][0], torch.Tensor):
        num_ax = len(dataset[0][0].shape)
        assert num_ax in [1, 2]
        task = 'regression' if num_ax == 2 else 'completion'
        inputs, targets = dataset
        tensor_input = isinstance(inputs, torch.Tensor)
    else:
        raise NotImplementedError

    # Loop until reps is 0
    while reps > 0:
        # For tensor recovery, just give the target tensor
        if task == 'recovery' or batch_size < 0:
            yield dataset

        # For regression, return minibatches of (input, target) data
        elif task in ['regression', 'completion']:
            ind = 0
            while ind < len(dataset[1]):
                inp = [ip[ind: ind+batch_size] for ip in inputs]
                if tensor_input or task == 'completion':
                    inp = torch.stack(inp)
                tar = targets[ind: ind+batch_size]
                ind += batch_size
                yield inp, tar
        reps = reps - 1
    return












### OLDER CODE ###

def get_repeated_Indices(list_of_indices):
    #input: string of indices for the tensors. 
    #output: string of repeated indices. code takes in indices and output only the repeated indices
    #Ex: Suppose tensor A has index ijkp, and tensor B has index klpm.
    #then get_repeated_Indices('ijkp', 'klpm') will return the following string: 'kp'
    myList = list_of_indices
    #convert List to string
    myString =''.join(myList)
    #break the string into indivual list of characters ex.  'abc' ->['a','b', 'c']
    myList = list(myString)
    #get the repeated frequencies of each indices
    my_dict = {i:myList.count(i) for i in myList}
    
    repeatedList = []
    for item in my_dict:
        if my_dict[item] > 1:
            repeatedList.append(item)
    return repeatedList

def remove_Repeated_indices(List_of_indices):
    #inputs: tensor indices in the form of string
    #output: string of non repeated indicies
    #Ex: remove_Repeated_indices('abc', 'cde')
    #output of the example would be: 'abde'
    
    myList = List_of_indices 
    #turn myList into String: Ex: ['abc','cde'] -> 'abccde'
    myString = ''.join(myList)
    #turn back into lists again: Exp: from 'abccde' -> ['a','b','c','c','d','e']
    myList = list(myString)
    repeated_indices = get_repeated_Indices(List_of_indices)
    #print('the repeated list of indices are:', repeated_indices)
    unique_indices = []
    #now we remove repeated indices from myList
    for item in myList:
        if item not in repeated_indices:
            unique_indices.append(item)
    uniqueString = ''.join(unique_indices)   
    return uniqueString

# TODO: This is a fantastic use of einsum, but should probably be rewritten
#       for two reasons (even though it breaks my heart):
#   (1) If the entire contraction of the tensor network is specified through
#       a single einsum string, we can only handle 52 (=2*26) different
#       edges, which means at most 9 nodes in our network. This might be
#       sufficient, but situations with 10 nodes could definitely still pop
#       up, and we don't want to be limited by this.
#   (2) We still need to generate indxList in order to use this function,
#       and indxList is currently written by hand. This isn't possible when
#       we have a variable number of nodes in our network.
def einSum_Contraction(tensorList, indxList):  #<----should rename this to einSum_Contraction to replace old code
    #Purpose: this function takes a list of tensors, and list of indices, and indix to contract and uses einstien summation to perform contraction
    #ex: tensorList = [tensor1, tensor2, tensor3]
    #indxList   = [indx1, indx2, indx3]
    myList = []
    uniqueIndices = remove_Repeated_indices(indxList)
    inputIndices = [indxList]
    N = len(indxList)
    #myList = [indx1, ',',indx2,',',indx3,'->', uniqueIndices] 
    for i in range(N - 1):
        myList.append(indxList[i])
        myList.append(',') 
    myList.append(indxList[N-1])
    myList.append('->')
    myList.append(uniqueIndices)
    #convert myList to a string: i.e.  [indx1, ',',indx2,',',indx3,'->', uniqueIndices]  - >'ijk,klm,mjp->ilp'
    myString = ''.join(myList)
    #print('myString = ', myString)
    C = torch.einsum(myString, tensorList)
    return C

def padTensor(tensor, pad_axis):
    #this is for the discrete optimization
    #this function takes a tensor and append an extra dimension of ~ zeros along the specified axis (we call the pad axis)
    if pad_axis == -1:
        return tensor #don't pad anything
    tensorShape = list(tensor.shape)
    tensorShape[pad_axis] = 1  #increase the dimension up by 1
    zerosPad = torch.rand(tensorShape) *1e-6  #pad with values approx. equal to zero
    padded_tensor = torch.cat([tensor, zerosPad], pad_axis)
    #print('padded_tensor.shape = ', padded_tensor.shape)
    #print('padded_tensor function output = ', padded_tensor)
    return padded_tensor

def increaseRank(Tensor1, Tensor2, indx1, indx2):
    # The indx 1 and index2 represents the indices for tensor 1 and 2 respectively. 
    #There is only one repeated index in the list (indx1, indx2). The repeated index represents the shared edge between
    #the two tensors. For ex: ijkl, lmno
    alpha = get_repeated_Indices([indx1, indx2])
    if len(alpha) != 0 :
        #convert alpha to string
        alpha = ''.join(alpha)
        # find the position of the repeated index alpha in indx1 and indx2
        padAxes1 = indx1.index(alpha)
        padAxes2 = indx2.index(alpha)  
        Tensor1 = padTensor(Tensor1, padAxes1)
        Tensor2 = padTensor(Tensor2, padAxes2)
    return  Tensor1, Tensor2

def Tensor_Generator(TensorDimension):
#input: desired target tensor dimension in the form of a list. Ex: input d1xd2xd3 as [d1, d2, d3]
#output: target tensor with random entries drawn from a normal distribution of mean=0 and variance=1
    Tensor = torch.randn(TensorDimension)
    return Tensor

def getOneData_point(W):
# W is input tensor
#X are drawn from a normal distribution of mean=0 and variance=1 
#output: Xi, yi  = targetTensor * X
    indxList = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    Xi = Tensor_Generator(W.shape) #generate Tensor Xi with same shape as input tensor W
    for j in range(len(Xi.shape)):
        indxList.append(alphabet[j])
        indx = ''.join(indxList)
    tensorList = [W, Xi]
    #W and X have same index so that yi is a scaler
    yi = einSum_Contraction([W,Xi], [indx, indx])
    return Xi, yi, indx  #W and Xi shares the same indx

def data_Set_Generator(tensor, N):
    #N = number of training data you want
    #tensor is usually the target tensor 
    Xi_set = []
    yi_set = []
    for i in range(N):
        Xi, yi, indx = getOneData_point(tensor)
        Xi_set.append(Xi)
        yi_set.append(yi)
    return Xi_set, yi_set,indx  #tensor and Xi shares the same indx

def getYi_set(Xi_set, tensor, indxList):
    #this function generates a N number of yi's using yi = tensor*Xi_set[i]. 
    #The input tensor is usually either the approx. tensor or the target tensor.
    #indxList = [indx_Xi, indx_tensor]
    yi_set = []
    N = len(Xi_set)      #is there are N elements of Xi in Xi_set, there will be N elements of yi in yi_set
    for i in range(N):
        yi = einSum_Contraction([tensor,Xi_set[i]], indxList)
        yi_set.append(yi)
    return yi_set

def innerProduct_Tensor(T,A):
    #input: inner product of two tensors T and A of same dimension equals the sum of the product of their entries 
    #covert T, and A tensors in to 1 D tensors
    T = T.view(1,-1)   #convert tensor  to 1D
    T = T.squeeze()    #squeeze out any extra dimension
    A = A.view(1,-1)
    A = A.squeeze()
    #perform inner product of two 1D tensors
    yi = sum(torch.mul(T,A))
    return yi
    

# TODO: Code should work for an arbitrary number of cores and edges in the
#       tensor network, not just for 4
def printRank(TensorList):
    [r5,r6,r7,r8] = (TensorList[4].shape) #tensor G dimension
    [r4,d4,r3,r5] = (TensorList[3].shape) #tensor D dimension
    [r2,d1,r1,r7] = (TensorList[0].shape) #tensor A dimension
    #print(' [r1, r2, r3,r4,r5,r6,r7,r8] = ', '[', r1, ',', r2, ',', r3, ',', r4, ',', r5,',', r6, ',', r7, ',', r8, ']')
    r = [r1,r2,r3,r4,r5,r6,r7,r8]
    return r

# TODO: Code should work for an arbitrary number of cores and edges in the
#       tensor network, not just for 4
def getNumParams(r):
    #r = list of ranks of a tensor
    numParam = d1*r[0]*r[6]*r[1] + d2*r[0]*r[7]*r[3]+ d3*r[1]*r[5]*r[2] + d4*r[3]*r[4]*r[2] + r[4]*r[5]*r[6]*r[7]
    return numParam

# TODO: Code should work for an arbitrary number of cores and edges in the
#       tensor network, not just for 4
def getNumParams_stoch(r):
    #r = [r0,r1,r2,r3,...,r7] 
    #we want to compuate the dimension of the block we start with. For exmape
    #we started with 4 order tensor, which can be approximated by 5 tensors
    #below computes the number of params for each of the tensor
    numParam1 = d1*r[0]*r[6]*r[1]
    numParam2 = d2*r[0]*r[7]*r[3]
    numParam3 = d3*r[1]*r[5]*r[2]
    numParam4 = d4*r[3]*r[4]*r[2]
    numParamCore = r[4]*r[5]*r[6]*r[7] 
    numParamList = [numParam1, numParam2, numParam3, numParam4, numParamCore]
    return numParamList

def computeLoss_Regression(targetData, approxTensor):
    # Unpack targetData
    yi_set, Xi_set = targetData

    #N is total number of training/test data
    sum = 0
    N = len(yi_set)
    for i in range(N):
        # y_approx = innerProduct(ApproxTensor, X_i)
        y_approx = innerProduct_Tensor(approxTensor, Xi_set[i])
        loss = (yi_set[i] - y_approx)**2
        sum = sum + loss
    #divide by N to average the squared error of the cost function  so that the cost function doesn't depend on the number 
    #of elements in the training set.
    total_loss = 1/(2*N)*sum 
    return(total_loss)

def computeLoss_Factorization(targetTensor, approxTensor):
    cost = torch.norm(approxTensor - targetTensor, 'fro') 
    return(cost)

def get_RandomSeqence(seqLength,d):
    #the function generates a random sequence of length 'seqLength' with range from 1 to range d.
    #For examplae, if we have 8 ranks, then seqLength. Code will generate 8 random numbers in the range between 
    #1 to d inclusive.
    r = []
    for i in range(seqLength):
        r.append(random.randint(1,d))
    return r

def get_Next_randomEdge(r_list):
    #this function is for random walk
    #input: r_list = [r0,r1,...] 
    nextDirection = random.randint(0, len(r_list)-1)
    print('nextDirection = ', nextDirection)
    r_list[nextDirection] = 1 + r_list[nextDirection]
    return r_list

def solve_Continuous(targetData, tensorList, indxList, iterNum, lossFun,
                     hyperparams):
#input: list of tensors and their corresponding indices
#Goal: The purpose of this function is to solve the innerloop of the optimization for the problem
    len_Tensor = len(tensorList)
    len_Indx   = len(indxList)
    for i in range(len_Tensor):
        tensorList[i] = tensorList[i].detach()
        tensorList[i].requires_grad = True
    
    # Unpack hyperparameters
    lr = hyperparams['lr']
    momentum = hyperparams['momentum'] if 'momentum' in hyperparams else None
    
    #defines a SGD optimizer to update the parameters
    #optimizer = optim.SGD(tensorList lr = 0.001, momentum=0.2)
    optimizer = optim.Adam(tensorList, lr=0.009)
    #initialize parameters      
    LostList = []              #use this to plot the lost function
    for i in range(iterNum):
        optimizer.zero_grad()
        tensor_approx = einSum_Contraction(tensorList, indxList)
        #loss_fn = computeLoss(tensor_approx, target_Tensor)   # this is for tensor decomp: ||W_target - W_approx||_Fˆ2
        loss_fn = lossFun(targetData, tensor_approx)
        loss_fn.backward()
        optimizer.step()                # the new A,B,C will be A_k+1,B_k+1, C_k+1 after optimizer.step 
        LostList.append(float(loss_fn))
    return tensorList, indxList, LostList
