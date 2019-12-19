# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:46:57 2019

@author: Mysia
"""

import numpy as np
import pandas as pd
import random
import getIrisData2 as datas
import sigm
import matplotlib.pyplot as plt
import nnloss


graph = np.zeros((2, 10))
l=0

trainingSet, testSet, y1, y2 = datas.getIrisData2()

# train data
input = trainingSet
# train labels
groundTruth = y1
batchSize=5
#Learning coefficient
coeff = 0.1
#Number of learning iterations
iterations = 1000
random.seed()
inputLength=input.shape[0]
hiddenN=5; #  nodes of hidden layer1
outputN=groundTruth.shape[1]
num_layers=3
tol=0.1

stack_bias=[[hiddenN+1/2],[1]]
stack_weights=[[hiddenN+1/2],[1]]

for k in range(0,num_layers-1,1):
    if(k==1):
        stack_weights[k]=np.random.rand(hiddenN,inputLength)
        stack_bias[k]=np.random.rand(hiddenN,1)
        
    elif(k==num_layers-1):            
        stack_weights[k] = np.random.rand(outputN,hiddenN)
        stack_bias[k] = np.random.rand(outputN,1)
    else:
        stack_weights[k] = np.random.rand(hiddenN,hiddenN)
        stack_bias[k] = np.random.rand(hiddenN,1)
        
outputStack1 = np.zeros((5, 1))
outputStack2 = np.zeros((5, 1))
outputStack3 = np.zeros((3, 1))

gradStack_epsilon1 = np.zeros((5, 1))
gradStack_epsilon2 = np.zeros((3, 1))

inputs=np.zeros((5,1))
p = np.zeros((5, 1))
gradStackEpsilon=[[num_layers],[1]]

for i in range(iterations):
    error=0
    for j in range(0,y1.shape[0],batchSize):
        data=input[:,j:j+batchSize]
        labels=groundTruth[j:j+batchSize,:]
        cost=0
        for kk in range(batchSize):
             inputs[0:5,0]=data[:,kk]      
             outputStack1=inputs
             # forward propagation
             
             outputStack2 = np.subtract(np.dot(stack_weights[0],outputStack1),stack_bias[0])
             outputStack2 = sigm.sigm(outputStack2)
             outputStack3 = np.subtract(np.dot(stack_weights[1],outputStack2),stack_bias[1])
             outputStack3 = sigm.sigm(outputStack3)
                 
#
             # backward propagation
             p[:,0] = outputStack3[:,0]
             epsilon = nnloss.nnloss(labels[kk: kk+1, :].T, p, 1)
             cost += nnloss.nnloss(labels[kk: kk+1, :].T, p, 0)
             error+=cost
             
             if j == 0:
                    gradStack_epsilon2 = np.multiply(np.multiply(outputStack3, 1-outputStack3), epsilon)
                    epsilon = np.dot(stack_weights[0].T, gradStack_epsilon2)
                    gradStack_epsilon1 = np.multiply(np.multiply(outputStack2, 1-outputStack2), epsilon)
                    epsilon = np.dot(stack_weights[1].T, gradStack_epsilon1)
             else:
                    gradStack_epsilon2 += np.multiply(np.multiply(outputStack3, 1-outputStack3), epsilon)
                    epsilon = np.dot(stack_weights[0].T, gradStack_epsilon2)
                    gradStack_epsilon1 += np.multiply(np.multiply(outputStack2, 1-outputStack2), epsilon)
                    epsilon = np.dot(stack_weights[1].T, gradStack_epsilon1)

        gradStack_epsilon2 = gradStack_epsilon2 / batchSize
        gradStack_epsilon1 = gradStack_epsilon1 / batchSize


        stack_weights[0] += np.multiply(coeff, np.dot(gradStack_epsilon1, outputStack1.T))
        stack_bias[0] += np.multiply((coeff*(-1)), gradStack_epsilon1)
        stack_weights[1] += np.multiply(coeff, np.dot(gradStack_epsilon2, outputStack2.T))
        stack_bias[1] += np.multiply((coeff*(-1)), gradStack_epsilon2)

        cost/=batchSize
        error += cost

#Plotting the graph
        
#        if (i%100) == 0:
#            graph[0, l] = i
#            graph[1, l] = error
#            l += 1
#        if abs(error)<tol:
#            break
#        
#    plt.plot(graph[0, 0: 10], graph[1, 0: 10], 'b*-')
#    plt.axis([0, 1000, 0, 100])
#    plt.show()
##    
        
#      % Update weights by   
#      % delta = coeff*epsilon*x   
#      % And use the new weights to repeat process.
        
np.save('stack1_b', stack_bias[0])
np.save('stack1_w', stack_weights[0])
np.save('stack2_b', stack_bias[1])
np.save('stack2_w', stack_weights[1])

testSet = np.load('testSet.npy')
y2 = np.load('y2.npy')

#%test the code
input = testSet
tol=0.1
groundTruth = y2
out = np.zeros([y2.shape[0],y2.shape[1]])

count = 0

for j in range(y2.shape[0]):
    inputs[:, 0] = input[:, j]
    outputStack1 = inputs
    
    #forward propagation
    outputStack2 = np.subtract(np.dot(stack_weights[0], outputStack1), stack_bias[0])
    outputStack2 = sigm.sigm(outputStack2)
    
    outputStack3 = np.subtract(np.dot(stack_weights[1], outputStack2), stack_bias[1])
    outputStack3 = sigm.sigm(outputStack3)
    
    out[:5] = outputStack3
    truth = groundTruth[j, :]
    o = out[:5]
    epsilon = np.subtract(truth, o)
    err = np.sum(np.power(epsilon, 2))
    if err<tol:
        count+=1

acc = (count/(out.shape[0])) * 100
print('accuracy of system: ', acc)



