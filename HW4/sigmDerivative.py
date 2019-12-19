# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:37:06 2019

@author: Mysia
"""
import numpy as np

def sigmDerivative(x):
    return np.multiply(x, 1-x)