# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:25:59 2019

@author: Mysia
"""
import scipy.io
import numpy as np

irisData = scipy.io.loadmat('irisData.mat')
stack = scipy.io.loadmat('stack.mat')
testSet = scipy.io.loadmat('testSet.mat')
trainingSet = scipy.io.loadmat('trainingSet.mat')
y1 = scipy.io.loadmat('y1.mat')
y2 = scipy.io.loadmat('y2.mat')

#Arrive all datas in irisData dictionary
#for key,val in irisData.items():
#    print (key, "=>", val)

#irisDatas=irisData.keys()
#print (irisDatas)


def getIrisData():
    
    for X in irisData:
        irisDatas = irisData['X']
            #   print (irisDatas)
            #    
            #
            #itemas=irisData.items()
            #print(itemas)
            #
            
    trainingSet=np.zeros((5,120),'float64')
    testSet=np.zeros((5,30),'float64')
    y1=np.zeros((120,3),'float64')
    y2=np.zeros((30,3),'float64')
            
    x=np.transpose(irisDatas)
            
    trainingSet[0:4,0:40] = x[:,0:40]
    trainingSet[0:4,41:80] = x[:,51:90]
    trainingSet[0:4,81:120] = x[:,101:140]
    trainingSet[4,:]=1
            
    testSet[0:4,0:10] = x[:,40:50]
    testSet[0:4,11:20] = x[:,91:100]
    testSet[0:4,21:30] = x[:,141:150]
    testSet[4,:]=1
            
    for i in range(0,40,1):
        y1[i,:]=[1,0,0]
        y1[i+40,:]=[0,1,0]
        y1[i+80,:]=[0,0,1] 
                
    for i in range(0,10,1):
        y2[i,:]=[1,0,0]
        y2[i+10,:]=[0,1,0]
        y2[i+20,:]=[0,0,1]
        
    np.save('trainingSet', trainingSet)
    np.save('testSet', testSet)
    np.save('y1', y1)
    np.save('y2', y2)
        
    return trainingSet, testSet, y1, y2    