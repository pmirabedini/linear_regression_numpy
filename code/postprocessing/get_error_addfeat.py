#!usr/bin/python3.6
  
"""
Created on Sat Jan 2 2021

@author: Pegah Mirabedini
"""

"""
you need to define the train and test datasets, and preprocess them if necessary.
trainX: train input: shape of (number of samples, number of features)
trainY: train target: shape of (number of samples,)
nfeat: number of features
"""

#import libraries:
import numpy as np
import matplotlib.pyplot as plt

def indx_sorted_arr(arr,sorted_arr):
    return [arr.index(x) for x in sorted_arr]


def get_error_addfeat(trainX,trainY,testX,testY,sorted_feats,nfeat,nadd):
    
    test_errs = []
    train_errs = []
    
    for f in sorted_feats[0:nfeat]:
        Pr = predict(trainX[:,sorted_feats[0:f+nadd+1]],trainY,testX[:,sorted_feats[0:f+nadd+1]],testY)
        _, _, train_err, test_err = Pr
        train_errs.append(train_err)
        test_errs.append(test_err)
        
    return test_errs, train_errs