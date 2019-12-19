# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:36:02 2019

@author: Mysia
"""
import numpy as np
import cv2
import os,sys
from scipy import misc
from os import listdir,makedirs
from os.path import isfile,join
from PIL import Image
import glob
from sklearn.utils import shuffle

        


#sigmoid function 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#derivative of sigmoid function 
def deriv_sigmoid(x):
    return (sigmoid(x)) * (1 - (sigmoid(x)))
 
  
#train perceptron
def trainPerceptron(inputs, t, weights, rho, iterNo):
   
    weight_sum=0
    
    for j in range(0,100,1):
        inputs, t = shuffle(inputs, t)
        
        for i in range(0,100,1):
        
            #feed forward
            x=inputs[i,:]
            y=np.dot(x,weights)
            o=sigmoid(y)
            
            #feed backward
            dw=rho*(t[i]-o)*deriv_sigmoid(o)*x
            weights+=dw
            
        
        
        #First try
#        #dot product inputs and weights
#        weights=np.dot(inputs[0],np.transpose(weights))
#        #then apply thw sigmoid function
#        xx=sigmoid(weight_sum)
#        
#         #find the errors (e) and return errors as weights
#        e=(t[0]-xx)
#        weights+=(rho*deriv_sigmoid(xx)*inputs[0])
        
    return weights

def testPerceptron(sample_test, weights):
    
    
    
    x=sample_test
    y=np.dot(x,weights)
    o=sigmoid(y)
    return o
        #First Try
#    weight_sum=0
#     #dot product sample_test and weights
#    weight_sum=np.dot(sample_test,np.transpose(weights))
#    
#    #then apply thw sigmoid function
#    xx=sigmoid(weight_sum)
#    #    return xx
        
videoPath1='./train/cannon'
class1=glob.glob(os.path.join(videoPath1,'*.jpg'))
    

videoPath2='./train/cellphone'
class2=glob.glob(os.path.join(videoPath2,'*.jpg'))

filenames=[]
for filename in class1:
    filenames.append(filename)
    
for filename in class2:
    filenames.append(filename)
        
inputs = np.ones((100,16385),'float32')
weights=np.random.rand(16385)/100
t=np.ones((100,1),'float32')
rho=0.001
iterNo=1000

t[0:42]=0
t[43:100]=1

for i in range(0,100,1):
    filename=filenames[i]
    img=misc.imread(filename,'L')
    img/=255
    img=cv2.resize(img, dsize=(128,128),interpolation=cv2.INTER_CUBIC)
    inputs[i,0:16384]=img.flatten()
    
weights=trainPerceptron(inputs,t,weights,rho,iterNo)

sample_test=np.ones((1,16385),'float32')
filename='./test/cannon/image_0021.jpg'
img=misc.imread(filename,'L')
img/=255
img=cv2.resize(img,dsize=(128,128),interpolation=cv2.INTER_CUBIC)
inputs[0,0:16384]=img.flatten()




sonuc=testPerceptron(sample_test,weights)
if(sonuc<0,5):
    print ('Object-1 belongs to class1')
else:
    print ('Object-1 belongs to class2')
    

sample_test=np.zeros((1,16385),'float32')
filename='./test/cellphone/image_0001.jpg'
img=misc.imread(filename,'L')
img/=255
img=cv2.resize(img,dsize=(128,128),interpolation=cv2.INTER_CUBIC)
inputs[0,0:16384]=img.flatten()

sonuc2=testPerceptron(sample_test,weights)
if(sonuc2<0,5):
    print ('Object-2 belongs to class2')
else:
    print ('Object-2 belongs to class1')

