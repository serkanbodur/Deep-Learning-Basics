# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:33:17 2019

@author: Mysia
"""
import numpy as np
def nnloss(x,t,dzdy):
    
    instanceWeights=np.ones(len(x))
    res=x-t
    
    if(dzdy==0):
        y = (1/2) * instanceWeights * np.power(res,2) 
    else:
        y = res
        
    return y