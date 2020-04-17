#!/usr/bin/env python
# coding: utf-8

# In[1]:


#see notes in blue binder for documents of this. 
#code was tested on three tensors: A-B-C
import sys
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from core_code import (einSum_Contraction, increaseRank, printRank, getNumParams,
                       computeLoss_Factorization, solve_Continuous)


# In[2]:


# TODO: Add function to core_code that initializes a random tensor with
#       prescribed input dimensions and ranks

#generate target tensor of rank 1
X =  torch.Tensor([[[1]],[[2]]])
Y =  torch.Tensor([[[3],[4],[5]]])
Z =  torch.Tensor([[[6,7,8,9]]])

d1 = 2
d2 = 3
d3 = 4
r1 = 2
r2 = 3
r3 = 2

X = torch.rand(d1, r3, r1)
Y = torch.rand(r1, d2, r2)
Z = torch.rand(r2, r3, d3)
print(X.shape)
print(Y.shape)
print(Z.shape)

indx0 = 'ijk'
indx1 = 'klm'
indx2 = 'mjp'

target_Tensor = einSum_Contraction([X, Y, Z], [indx0, indx1, indx2])
print('shapeTargetTensor=', target_Tensor.shape)


# In[3]:


# TODO: Use the function mentioned in the above cell to initialize our tensor network

#initilize
r1 = 1
r2 = 1
r3 = 1
#generate at random target tensor
X_approx0 = torch.rand(d1,r3,r1)*1
X_approx0 = X_approx0/torch.norm(X_approx0, 'fro') 
Y_approx0 = torch.rand(r1,d2,r2)*1
Y_approx0 = Y_approx0/torch.norm(Y_approx0, 'fro') 
Z_approx0 = torch.rand(r2,r3,d3)*1
Z_approx0 = Z_approx0/torch.norm(Z_approx0, 'fro') 

##this intialize tensor close to target tensor. This is to confirm the loss function is low as eplected
#noise = 0.000000000001
#X_approx0 =  torch.Tensor([[[1]],[[2]]]) + torch.rand(2,1,1)*noise  #need to check it is from normal distribution
#Y_approx0 =  torch.Tensor([[[3],[4],[5]]])+ torch.rand(1,3,1)*noise
#Z_approx0 =  torch.Tensor([[[6,7,8,9]]])+ torch.rand(1,1,4)*noise

indx0 = 'ijk'
indx1 = 'kmn'
indx2 = 'njp'
indxList = [indx0, indx1, indx2]


# In[4]:


# Set hyperparameters for the experiment (anything you want to feed to the Pytorch optimizer)
hyperparams = {'lr': 0.009,
               'some_other_param': 2}


# In[5]:


# TODO: Package up code for the greedy optimization loop into a standalone function
#       in core_tools, where the loss function and target data (among other things) 
#       are inputs

#Here we added a third loop. Each iteration samples a different grid point and its nearby neighbours. It computes the continuous
#opt for different rank combinations and stores the one with the least minimum.
A = X_approx0
B = Y_approx0
C = Z_approx0

indxList = [indx0, indx1, indx2]
TensorList = [A, B, C]
TensorList_temp = [A, B, C] #TensorList[:]
iterNum=500
Lost_star = 5
check = 1
maxParam = d1*d2*d3
numParam = -1
paramKey = -1

for k in range(5):
    if paramKey == 1:
        break
    for i in range(len(TensorList_temp)):
        if paramKey == 1:
            break
        for j in range(i+1,len(TensorList_temp)):
            if paramKey == 1:
                break
            print(i,j)
            #increase the ranks of the tensors
            [TensorList_temp[i],TensorList_temp[j]] = increaseRank(TensorList_temp[i], TensorList_temp[j],  indxList[i], indxList[j])
            
            #check num of paramters for the newly updated ranks
            [r1_t,r2_t,r3_t] = printRank(TensorList_temp)
            numParam_temp = getNumParams(r1_t,r2_t,r3_t)
            print('num of parameters for recently updated ranks= ', numParam_temp)
            if numParam_temp > maxParam:
                paramKey = 1
                print('Max number of parameters exceeded. Current Param = ', numParam_temp, 'and max Param allowed = ', maxParam)
                print('program finish ')
                break
            #solve continuous part
            [TensorList_temp, indxList, LostList] = solve_Continuous(target_Tensor, TensorList_temp, indxList, iterNum, 
                                                                     computeLoss_Factorization, hyperparams)
            print('Currently evaluating rank:')
            printRank(TensorList_temp)
            
            #store the optimal value for a given point around its neighbour
            if Lost_star > LostList[-1]: #avgLost:  #When have time, take the average of 10 elements as opposed to the min. of last elementLostList[-1]:  #When have time, take the average of 10 elements as opposed to the min. of last element
                    indx_star = [i,j]
                    TensorList_star = TensorList_temp[:]
                    indxList_star = indxList[:]
                    print('Lost* updated: Lost_star previous =', Lost_star, 'Lost_star current =', LostList[-1])
                    Lost_star = LostList[-1]  
                    print('current Best rank:')
                    [r1,r2,r3] = printRank(TensorList_star)
                    numParam = getNumParams(r1,r2,r3)
                    #numParam = d1*r1*r3 + r1*d2*r2 + r2*r3*d3
                    #print('d1*r1*r3 = ', d1*r1*r3)
                    #print('r1*d2*r2', r1*d2*r2)
                    #print('r2*r3*d3', r2*r3*d3)
                    print('total number of parameters for best chosen ranks = ', numParam)
                    check = -1
            elif Lost_star <= LostList[-1]:
                print('Prev Loss = ', Lost_star, 'Current Loss = ', LostList[-1])
                print('Loss previous is less than current, so no rank update is made at this iteration')
                print('total number of parameters for best chosen rank = ', numParam)

            
            #Reset TensorList_temp to continue with greedy at another point
            TensorList_temp = TensorList[:] #set back to previous position to continue with greedy search
            print('***************************************************************')  
    print('****************************moving to another grid point ***********************************')
    #update parameters   
    #print('TensorList_star[0] = ', TensorList_star[0])
    #print('TensorList_star[1] = ', TensorList_star[1])
    #print('TensorList_star[2] = ', TensorList_star[2])
    TensorList_temp = TensorList_star[:]    #everything is behaving as expected
    #print('TensorList_temp[0] After = ', TensorList_temp[0])
    #print('TensorList_temp[1] After = ', TensorList_temp[1])
    #print('TensorList_temp[2] After = ', TensorList_temp[2])
    #print('indxList_star[2] before = ', indxList_star[2])
    indxList = indxList_star[:]     # don't really need to update this cause these don't really change
    #print('indxList[2] after = ', indxList[2])
    #print('TensorList_star[0] =',  TensorList_star[0])
    TensorList  = TensorList_star[:]  #everything is behaving as expected
    #print('TensorList[0] =',  TensorList[0])
    


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
plt.figure()
plt.plot(LostList)
print('loss function approaches =', LostList[-1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




