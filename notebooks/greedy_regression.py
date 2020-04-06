#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import torch
import numpy as np
from numpy.linalg import norm
import torch.optim as optim
import matplotlib.pyplot as plt
import random

sys.path.append('..')
from core_code import (einSum_Contraction, increaseRank, data_Set_Generator, 
                      getYi_set, printRank, getNumParams, computeLoss_Regression, 
                      get_RandomSeqence, get_Next_randomEdge, solve_Continuous)


# In[2]:


# TODO: Add function to core_code that initializes a random tensor with
#       prescribed input dimensions and ranks

#generate 4-order target tensor
#see ipad for supplementary notes on this tensor and its diagram

#With Core tensor: 5 nodes
#****it will be interestingn to set r5 to r8 to get tucker structure and run the code to see how 
# the structure compares to tucker decomposition. Tucker decomposition could become another based cased
d1 = 3
d2 = 3
d3 = 3
d4 = 3
d5 = 3
r0 = 2
r1 = 3
r2 = 4
r3 = 3
r4 = 2
r5 = 3
r6 = 2
r7 = 2

noise = 1e-6
#generate at random target tensor

A = torch.rand(r1,d1,r0,r6) # + torch.rand(r1,d1,r0,r6)*noise
B = torch.rand(r0,d2,r3,r7) # + torch.rand(r0,d2,r3,r7)*noise
C = torch.rand(r1,d3,r2,r5) # + torch.rand(r1,d3,r2,r5)*noise
D = torch.rand(r3,d4,r2,r4) # + torch.rand(r3,d4,r2,r4)*noise 
G = torch.rand(r4,r5,r6,r7) # + torch.rand(r4,r5,r6,r7)*noise

indxA = 'kilc'
indxB = 'ljed'
indxC = 'khgb'
indxD = 'efga'
indxG = 'abcd'   #core tensor

indxList = [indxA, indxB, indxC, indxD, indxG]

target_Tensor = einSum_Contraction([A,B,C,D,G], [indxA, indxB, indxC, indxD, indxG])


#generate training and test sets
N=20
#p is the percentage of the total number of data N
p = int(np.floor(0.65*N))  #so is is 65% of the original data
[Xi_data, yi_data, indx] = data_Set_Generator(target_Tensor, N) 
Xi_train = Xi_data[0:p]
yi_train  = yi_data[0:p]
Xi_test = Xi_data[p:N+1]
yi_test = yi_data[p:N+1]


# In[3]:


# Set hyperparameters for the experiment (anything you want to feed to the Pytorch optimizer)
hyperparams = {'lr': 0.009,
               'some_other_param': 2}


# In[4]:


# TODO: Use the function mentioned in the above cell to initialize our tensor network

#following data are used for initializing Greedy method

#initilize data
r1 = 1
r2 = 1
r3 = 1
r4 = 1
r5 = 1
r6 = 1
r7 = 1
r8 = 1

#initialize tensor 
A_0 = torch.rand(r2,d1,r1,r7)
B_0 = torch.rand(r1,d2,r4,r8)
C_0 = torch.rand(r2,d3,r3,r6) 
D_0 = torch.rand(r4,d4,r3,r5) 
G_0 = torch.rand(r5,r6,r7,r8)

#target_Tensor = einSum_Contraction([A_0,B_0,C_0,D_0,G_0], [indxA, indxB, indxC, indxD, indxG])


TensorList = [A_0,B_0,C_0,D_0,G_0] 
TensorList_temp = [A_0,B_0,C_0,D_0,G_0] #TensorList[:]

#index list: we use the same indexlist as the ones defiined for target tensor


# In[5]:


# TODO: Package up code for the greedy optimization loop into a standalone function
#       in core_tools, where the loss function and target data (among other things) 
#       are inputs

############################### MAIN LOOP FOR GREEDY #####################

#Initialize data
iterNum=100   #500
Lost_star = 1e12  #set it to be any large number
check = 1
maxParam = d1*d2*d3*d4
numParam = -1
paramKey = -1
G = []

for k in range(5):
    if paramKey == 1:
        break
    for i in range(len(TensorList_temp)):
        if paramKey == 1:
            break
        for j in range(i,len(TensorList_temp)):
            if paramKey == 1:
                break
            if i==j:
                continue
            #print(i,j)
            #increase the ranks of the tensors
            [TensorList_temp[i],TensorList_temp[j]] = increaseRank(TensorList_temp[i], TensorList_temp[j],  indxList[i], indxList[j])            
            #check num of paramters for the newly updated ranks
           # rt = list of ranks: [r1_t,r2_t,r3_t,r4_t,r5_t,r6_t,r7_t,r8_t]
            rt = printRank(TensorList_temp)
            numParam_temp = getNumParams(rt)
            print('numParam_greedy=', numParam_temp)
            if numParam_temp > maxParam:
                paramKey = 1
                print('Max number of parameters exceeded. Current Param = ', numParam_temp, 'and max Param allowed = ', maxParam)
                print('program finish ')
                break
            #solve continuous part
            targetData = (yi_train, Xi_train)
            [TensorList_temp, indxList, LostList] = solve_Continuous(targetData, TensorList_temp, indxList, iterNum,
                                                                     computeLoss_Regression, hyperparams)
            printRank(TensorList_temp)
            #store the optimal value for a given point around its neighbour
            if Lost_star > LostList[-1]: 
                    indx_star = [i,j]
                    TensorList_star = TensorList_temp[:]
                    indxList_star = indxList[:]  
                    Lost_star = LostList[-1]  
                    r_greedy = printRank(TensorList_star)
                    print('r_greedy = ', r_greedy)
                    numParam = getNumParams(r_greedy )
                    check = -1
            elif Lost_star <= LostList[-1]:
                print('k = ', k, 'Prev Loss = ', Lost_star, 'Current Loss = ', LostList[-1])
                print('Loss previous is less than current, so no rank update is made at this iteration')
                print('total number of parameters for best chosen rank = ', numParam)
         
            #Reset TensorList_temp to continue with greedy at another point
            TensorList_temp = TensorList[:] #set back to previous position to continue with greedy search        
    #update parameters   
    TensorList_temp = TensorList_star[:]    #everything is behaving as expected
    indxList = indxList_star[:]     # don't really need to update this cause these don't really change
    TensorList_greedy   = TensorList_star[:]  #everything is behaving as expected
    G.append(Lost_star)
    


# In[ ]:


#GREEDY script continue
#TensorList contains decomposed block of tensors. We use einsum to combine them into one 
#big tensor Tapprox_greedy
Tapprox_greedy = einSum_Contraction(TensorList_greedy, indxList)  #TensorList_Greedy = TensorList
print('Tapprox_greedy.shape = ', Tapprox_greedy.shape)

#generate yi_approx = tensorApprox_star * X_i. Plot this set of yi_approx, with yi
indxlist_greedy = ['abcd', 'abcd']   # abcd =  = d1*d2*d3*d4 i.e. each alphabet represents each dimension ot target tensor
yi_approxSet_greedy = getYi_set(Xi_test, Tapprox_greedy, indxlist_greedy)
print('len(yi_approxSet_greedy) = ', len(yi_approxSet_greedy))


# In[ ]:


# plot for all the greedy loops
   #import matplotlib.pyplot as plt
plt.figure()
plt.plot(G, 'o')
plt.show()
print('greedy: G = ', G)


# In[ ]:


# TODO: Package up code for the stochastic optimization loop into a standalone function
#       in core_tools, where the loss function and target data (among other things) 
#       are inputs

#***********************STOCHASTIC APPROACH MAIN******************************************

#The following main program assigns random values to each of the 8 ranks and computes the continuos optimization
#part. It repeats 8 times (8 trials) and selects the trial with the smallest lost.

#initilize data
iterNum=100   #500
maxParam = d1*d2*d3*d4
#print('maxParam = ', maxParam) 
L = []
#generate sequence of random ranks between 1 to 8
numRank = 8
d=4  #highest dimension you want to explore
max_numShots = 15  #number of random trials you want to perform
dmax = max(d1,d2,d3,d4)
numShots = 0
LostList_prev = 1e12

#for i in range(numShots):
while numShots <max_numShots:
    r_stoch = get_RandomSeqence(numRank,d)
    #initialize tensor 
    A_0 = torch.rand(r_stoch[1],d1,r_stoch[0],r_stoch[6])
    B_0 = torch.rand(r_stoch[0],d2,r_stoch[3],r_stoch[7])
    C_0 = torch.rand(r_stoch[1],d3,r_stoch[2],r_stoch[5])
    D_0 = torch.rand(r_stoch[3],d4,r_stoch[2],r_stoch[4]) 
    G_0 = torch.rand(r_stoch[4],r_stoch[5],r_stoch[6],r_stoch[7])
    TensorList_temp = [A_0,B_0,C_0,D_0,G_0] #TensorList[:]   
    #the following if loop checks the number of elements of each A_0, ..., G_0
    #does not exceed the total number elements of target tensor
    #numParamList = getNumParams(r_stoch)
    #if maxParam > numParamList[0] and maxParam > numParamList[1] and maxParam > numParamList[2] and maxParam > numParamList[3] and maxParam > numParamList[4]:
    if maxParam > getNumParams(r_stoch):
        targetData = (yi_train, Xi_train)
        [TensorList_temp, indxList, LostList] = solve_Continuous(targetData, TensorList_temp, indxList, iterNum,
                                                                 computeLoss_Regression, hyperparams)
        L.append(LostList[-1])
        numShots += 1
        #the following if-loop stores the approx tensor that gives the smallest loss
        if LostList[-1] <LostList_prev:
            tensorApprox_star = TensorList_temp[:]
            LostList_prev = LostList[-1]
            r_Star = r_stoch       
print('Stochastic Method: best rank r = ', r_Star, 'with loss = ', LostList_prev)


# In[ ]:


#...STOCHASTIC SCRIPT CONTINUE
#tensorApprox_star consists of blocks of tensor. We take tensor product of all these
#blocks to get one big block: TensorApprox_Stoch
indxList = [indxA, indxB, indxC, indxD, indxG]  #use the exact same indxList as the ones above
TensorApprox_Stoch = einSum_Contraction([tensorApprox_star[0],tensorApprox_star[1],tensorApprox_star[2],tensorApprox_star[3],tensorApprox_star[4]], [indxA, indxB, indxC, indxD, indxG])

#generate yi_approx = tensorApprox_star * X_i. 
indxlist_stoch = [indx, indx]   # indx_Xi = indx_W = indx. 
yi_approxSet_stoch = getYi_set(Xi_test, TensorApprox_Stoch, indxlist_stoch)


# In[ ]:


# TODO: Package up code for the random walk optimization loop into a standalone function
#       in core_tools, where the loss function and target data (among other things) 
#       are inputs

#################################Random Walk(RW) main###############################
maxParam = d1*d2*d3*d4
iterNum = 100
#initialize the edges to 1
r_RW = [1,1,1,1,1,1,1,1]
numParam_RW = getNumParams(r_RW)
Lost_starRW = 1e12  #set it to be any large number
G_RW = []
while maxParam > numParam_RW:
    print('r_RW =', r_RW)
    r_RW = get_Next_randomEdge(r_RW)
    numParam_RW = getNumParams(r_RW)
    #initialize tensor 
    if maxParam > numParam_RW:
        A_0 = torch.rand(r_RW[1],d1,r_RW[0],r_RW[6])
        B_0 = torch.rand(r_RW[0],d2,r_RW[3],r_RW[7])
        C_0 = torch.rand(r_RW[1],d3,r_RW[2],r_RW[5])
        D_0 = torch.rand(r_RW[3],d4,r_RW[2],r_RW[4]) 
        G_0 = torch.rand(r_RW[4],r_RW[5],r_RW[6],r_RW[7])
        TensorList_RW = [A_0,B_0,C_0,D_0,G_0]
        targetData = (yi_train, Xi_train)
        [TensorList_RW, indxList, LostList_RW] = solve_Continuous(targetData, TensorList_RW, indxList, iterNum,
                                                                  computeLoss_Regression, hyperparams)
        G_RW.append(LostList_RW[-1])

print('Random Walk: numParam = ', numParam_RW, 'maxNumParam of target tensor = ', maxParam)
print('G_RW = ', G_RW)

#take the block tensors and combine them to form 1 big tensor: TensorApprox_RW 
indxList = [indxA, indxB, indxC, indxD, indxG]  #use the exact same indxList as the ones above
TensorApprox_RW = einSum_Contraction([TensorList_RW[0], TensorList_RW[1], TensorList_RW[2], TensorList_RW[3], TensorList_RW[4]], [indxA, indxB, indxC, indxD, indxG])
indxlist_RW = ['abcd', 'abcd']   # abcd =  = d1*d2*d3*d4 i.e. each alphabet represents each dimension ot target tensor
yi_approxSet_RW = getYi_set(Xi_test, TensorApprox_RW, indxlist_RW)


# In[ ]:


##PLOT FOR RANDOM WALK. X-AXIS = total number of steps and Y-axis = LostList[-1]
import matplotlib.pyplot as plt
plt.figure()
plt.plot(G_RW, 'og')
plt.show()


# In[ ]:


#we assume yi_actual = yi_train. 
#The following plot compares the approx yi using different the 3 different approach (greedy, stochastic, random walk)
# and compare the approx to the actual data
import matplotlib.pyplot as plt
plt.figure()
plt.plot(yi_approxSet_stoch, 'og')
plt.plot(yi_approxSet_greedy, 'ob')
plt.plot(yi_approxSet_RW, 'om')
plt.plot(yi_test, '*r')   #recall we treat yi_test as our true data
plt.title('Greedy, stochastic, randomWalk data vs actual data ',loc='center')
plt.show()

print('Legend')
print('red: actual data')
print('green: predicted data using stochastic')
print('blue: predicted data using greedy')
print('magenta: predicted data using random walk')


# In[ ]:


# Compute RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE_stoch = sqrt(mean_squared_error(yi_test,yi_approxSet))
print('RMSE_stoch = ',RMSE_stoch)

RMSE_greedy = sqrt(mean_squared_error(yi_test,yi_approxSet_greedy))
print('RMSE_greedy = ',RMSE_greedy) 

RMSE_RW = sqrt(mean_squared_error(yi_test,yi_approxSet_RW))
print('RMSE_randomWalk = ',RMSE_RW) 


# In[ ]:


#relative error
def getRelError(yi_actual,yi_pred):
# Both yi_actual,yi_pred are a set of elements
    summ = 0
    nm = 0
    N = len(yi_actual)
    for i in range(N):
        summ = summ + abs(yi_actual[i] - yi_pred[i])
        nm = nm + abs(yi_actual[i])
    return summ/nm*100 

RE_stoch = getRelError(yi_test,yi_approxSet_stoch)
print('stoch Method: % rel. error = ', RE_stoch)

RE_greedy = getRelError(yi_test,yi_approxSet_greedy)
print('Greedy Method: % rel. error = ', RE_greedy)

RE_RW = getRelError(yi_test,yi_approxSet_RW)
print('Random Walk Method: % rel. error = ', RE_RW)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




