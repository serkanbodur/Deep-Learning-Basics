# -*- coding: utf-8 -*-
"""
Created on Wed Nov  19 21:02:37 2019

@author: Mysia
"""



import os
import numpy as np
from scipy import misc
import cv2
import glob
 

#Read the images
videoPath1='./train/butterfly'
class1=glob.glob(os.path.join(videoPath1,'*.jpg'))

videoPath2='./train/chair'
class2=glob.glob(os.path.join(videoPath2,'*.jpg'))

videoPath3='./train/laptop'
class3=glob.glob(os.path.join(videoPath3,'*.jpg'))

#Create a file and merge all images in there
filenames=[]
for filename in class1:
    filenames.append(filename)
    
for filename in class2:
    filenames.append(filename)
    
for filename in class3:
    filenames.append(filename)

#Prepare inputs and labels
inputs = np.ones((231,16385),'float32')
labels=np.ones((231,1),'float32')

labels[0:90]=0
labels[91:151]=1
labels[152:231]=2

#Convert to image 128*128 format  Then 1*16384 vector
for l in range(0,231,1):
    filename=filenames[l]
    img=misc.imread(filename,'L')
    img/=255
    img=cv2.resize(img, dsize=(128,128),interpolation=cv2.INTER_CUBIC)
    inputs[l,0:16384]=img.flatten()

#merge train set and labels
x_and_y_train=np.concatenate([inputs,labels],axis=1)


#Read the test chair
chair_test=np.ones((1,16386),'float32')
videoTest1='./test/image_0001.jpg'
img=misc.imread(videoTest1,'L')
img/=255
img=cv2.resize(img,dsize=(128,128),interpolation=cv2.INTER_CUBIC)
chair_test[0,0:16384]=img.flatten()

#Read the test laptop
laptop_test=np.ones((1,16386),'float32')
videoTest2='./test/image_0007.jpg'
img=misc.imread(videoTest2,'L')
img/=255
img=cv2.resize(img,dsize=(128,128),interpolation=cv2.INTER_CUBIC)
laptop_test[0,0:16384]=img.flatten()

#Read the test butterfly
butterfly_test=np.ones((1,16386),'float32')
videoTest3='./test/image_0031.jpg'
img=misc.imread(videoTest3,'L')
img/=255
img=cv2.resize(img,dsize=(128,128),interpolation=cv2.INTER_CUBIC)
butterfly_test[0,0:16384]=img.flatten()

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return np.sqrt(distance)


#sorting of list till first k value
def sort_train(train,test_row,k_neighbour):
    a=list()
    for train_row in train:
        dist=euclidean_distance(test_row,train_row)
        a.append((train_row,dist))
    a.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k_neighbour):
    	neighbors.append(a[i][0])
    return neighbors

#find with knn
#We get neigbors after sorting
#Then find the nearest neighbors up to k value
def KNN(x_train, y_train, sample_test, k ):
    neigbors=sort_train(x_train,sample_test,k)
    outputs=[row[-1] for row in neigbors]
    y_train = max(set(outputs), key=outputs.count)
    return y_train

#If the k=7, The result of chair is wrong
    
print("When k values is 7")
k=7


mustbe0=KNN(x_and_y_train,labels,butterfly_test[0],k)
if mustbe0==0:
    print("This image is butterfly")
elif mustbe0==1:     
    print("Wrong Result.The image is found as chair.It should have been butterfly")
elif mustbe0==2:
    print("Wrong Result.The image is found as laptop.It should have been butterfly")


mustbe1=KNN(x_and_y_train,labels,chair_test[0],k)
if mustbe1==1:
    print("This image is chair")
elif mustbe1==0:
    print("Wrong Result.The image is found as butterfly.It should have been chair")
elif mustbe1==2:
    print("Wrong Result.The image is found as laptop.It should have been chair")
        

mustbe2=KNN(x_and_y_train,labels,laptop_test[0],k)
if mustbe2==2:
    print("This image is laptop")
elif mustbe2==0:
    print("Wrong Result.The image is found as butterfly.It should have been laptop")
elif mustbe2==2:
    print("Wrong Result.The image is found as chair.It should have been laptop")
    
#If the k=10,every results are true
print("When k values is 10")
ktrue=10


mustbe0=KNN(x_and_y_train,labels,butterfly_test[0],ktrue)
if mustbe0==0:
    print("This image is butterfly")
elif mustbe0==1:     
    print("Wrong Result.The image is found as chair.It should have been butterfly")
elif mustbe0==2:
    print("Wrong Result.The image is found as laptop.It should have been butterfly")


mustbe1=KNN(x_and_y_train,labels,chair_test[0],ktrue)
if mustbe1==1:
    print("This image is chair")
elif mustbe1==0:
    print("Wrong Result.The image is found as butterfly.It should have been chair")
elif mustbe1==2:
    print("Wrong Result.The image is found as laptop.It should have been chair")
        

mustbe2=KNN(x_and_y_train,labels,laptop_test[0],ktrue)
if mustbe2==2:
    print("This image is laptop")
elif mustbe2==0:
    print("Wrong Result.The image is found as butterfly.It should have been laptop")
elif mustbe2==2:
    print("Wrong Result.The image is found as chair.It should have been laptop")




