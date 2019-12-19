#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 19:00:01 2019

@author: Mysia
"""

import cv2
import numpy as np


def flip_vertical(image):
    newimg = np.zeros(image.shape, dtype=np.uint8)
    rows = image.shape[0]
    cols = image.shape[1]
    for r in range(rows):
        for c in range(cols):
            newimg[r,c] = image[rows-r-1,c]
    return newimg

def flip_horizontal(image):
    newimg = np.zeros(image.shape, dtype=np.uint8)
    rows = image.shape[0]
    cols = image.shape[1]
    for r in range(rows):
        for c in range(cols):
            newimg[r,c] = image[r,cols-c-1]
    return newimg
    
def rotate_left(image):
    rows = image.shape[0]
    cols = image.shape[1]
    newimg = np.zeros((cols, rows, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            newimg[c,r] = image[r,cols-c-1]
    return newimg

def rotate_right(image):
    rows = image.shape[0]
    cols = image.shape[1]
    newimg = np.zeros((cols, rows, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            newimg[c,r] = image[rows-r-1,c]
    return newimg

def downscale(image):
    rows = image.shape[0]
    cols = image.shape[1]
    nrows = rows//2
    ncols = cols//2
    newimg = np.zeros((nrows, ncols, 3), dtype=np.uint8)
    for r in range(nrows):
        for c in range(ncols):
            rr = r*2
            cc = c*2
            avg = np.zeros([3])
            avg = avg + image[rr,cc]
            avg = avg + image[rr,cc+1]
            avg = avg + image[rr+1,cc]
            avg = avg + image[rr+1,cc+1]
            avg = avg / 4
            newimg[r,c] = avg
    return newimg

    
img = cv2.imread("cat.png")
print(img.shape)

cv2.imshow("input", img)
cv2.waitKey()
cv2.imshow("flip_vertical", flip_vertical(img))
cv2.waitKey()
cv2.imshow("flip_horizontal", flip_horizontal(img))
cv2.waitKey()
cv2.imshow("rotate_left", rotate_left(img))
cv2.waitKey()
cv2.imshow("rotate_right", rotate_right(img))
cv2.waitKey()
cv2.imshow("downscale", downscale(img))
cv2.waitKey()