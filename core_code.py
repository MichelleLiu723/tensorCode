#!/usr/bin/env python
# coding: utf-8
import random
from functools import reduce

import torch
import numpy as np
import tensornetwork as tn
import torch.optim as optim
"""
### NOTES ON TENSOR NETWORK FORMAT ###

A tensor network is specified as a list of tensors, each of whose shape is
formatted in a specific way. In particular, for a network with n tensor
cores, the shape will be:

tensor_i.shape = (r_1, r_2, ..., r_i, ..., r_n, [batch]),

where r_j gives the tensor network rank (TN-rank) connecting cores i and j
when j != i, while r_i gives the dimension of the input to core i. This 
implies that tensor_i.shape[j] == tensor_j.shape[i], and stacking the 
shapes of all the tensors in order gives the adjacency matrix of the 
network (ignoring the diagonals).

The optional batch index allows for multiple networks to be processed in 
parallel. It's a little bit hackey getting batch inputs to work well with 
PyTorch and TensorNetwork, but such is life.

On occasion, the ranks are specified in the following triangular format:
     [[r_{1,2}, r_{1,3}, ..., r_{1,n}], [r_{2,3}, ..., r_{2,n}], ...
      ..., [r_{n-2,n-1}, r_{n-2,n}], [r_{n-1,n}]]
"""

# Set global defaults
tn.set_default_backend("pytorch")
torch.set_default_tensor_type(torch.DoubleTensor)


def contract_closed_network(tensor_list):
    """
    Contract a closed (no inputs) tensor network to get a scalar

    Args:
        tensor_list: List of (properly formatted) tensors that encodes the
                     closed tensor network

    Returns:
        scalar:      Scalar output giving the value of contracted network
    """
    pass

def contract_inputs(tensor_list, input_list):
    """
    Contract input vectors with open tensor network to get closed network

    Args:
        tensor_list: List of (properly formatted) tensors that encodes the
                     open tensor network
        input_list:  List of inputs for each of the cores in our tensor

    Returns:
        closed_list: List of tensors that encodes the closed tensor network
    """
    pass

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
    pass

def random_network(input_dims, ranks=1):
    """
    Initialize a tensor network with random (normally distributed) cores

    Args:
        input_dims:  List of input dimensions for each core in the network
        ranks:       Scalar or list of ranks connecting different cores.
                     For scalar inputs, all ranks will be initialized at
                     the specified number, whereas more fine-grained ranks
                     are specified in the following triangular format:
                     [[r_{1,2}, r_{1,3}, ..., r_{1,n}], [r_{2,3}, ..., r_{2,n}],
                      ..., [r_{n-2,n-1}, r_{n-2,n}], [r_{n-1,n}]]

    Returns:
        tensor_list: List of randomly initialized and properly formatted
                     tensors encoding our tensor network
    """
    def stdev_fun(shape):
        """Heuristic function which converts shapes into standard devs"""
        num_el = np.prod(np.array(shape))
        return np.sqrt(num_el)

    # Process input and convert input into core shapes
    num_cores = len(input_dims)
    if not hasattr(ranks, '__len__'):
        ranks = [[ranks] * e for e in range(num_cores - 1, 0, -1)]
    assert len(ranks) == num_cores - 1

    shape_list = unpack_ranks(input_dims, ranks)

    # Use shapes to instantiate random core tensors
    tensor_list = []
    for shape in shape_list:
        tensor_list.append(stdev_fun(shape) * torch.randn(shape))

    return tensor_list

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

def better_copynode(num_edges, dimension):
    """
    Return a list of small connected nodes which emulates a large CopyNode

    Args:
        num_edges: The number of dangling edges in the output, equivalent 
                   to the `rank` parameter of CopyNode
        dimension: The dimension of each of the edges of the output

    Returns:
        edge_list: List of edges of our composite CopyNode object
    """
    # For small numbers of edges, just use a single CopyNode
    if num_edges < 4:
        node = tn.CopyNode(rank=num_edges, dimension=dimension)
        return node.get_all_edges()

    # Initialize list of free edges with output of trivial identity mats
    input_node = tn.Node(torch.eye(dimension))
    edge_list, dummy_list = zip(*[input_node.copy().get_all_edges() 
                                  for _ in range(num_edges)])

    # Iteratively contract dummy edges until we have less than 4
    dummy_len = len(dummy_list)
    while dummy_len > 4:
        odd = dummy_len % 2 == 1
        half_len = dummy_len // 2

        # Apply third order tensor to contract two dummy indices together
        temp_list = []
        for i in range(half_len):
            temp_node = tn.CopyNode(rank=3, dimension=dimension)
            temp_node[1] ^ dummy_list[2 * i]
            temp_node[2] ^ dummy_list[2 * i + 1]
            temp_list.append(temp_node[0])
        if odd:
            temp_list.append(dummy_list[-1])

        dummy_list = temp_list
        dummy_len = len(dummy_list)

    # Contract the last dummy indices together
    last_node = tn.CopyNode(rank=dummy_len, dimension=dimension)
    [last_node[i] ^ dummy_list[i] for i in range(dummy_len)]

    return edge_list

### OLD CODE ###

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

def avgPoints(list1, num_points):
    #num_points = number of elements from list1 that you want to take the average of
    n = len(list1)
    avg = sum(list1[n-num_points:n])/num_points
    return avgPoints