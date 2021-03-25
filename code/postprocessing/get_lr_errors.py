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

def get_errors(trainX,trainY,testX,testY):
    
    """ 
    This function runs linear regression for each feature individually and store the train and test error.
    You need to run the lr.py script for your data first. 
    """
    
    err_train_all = []
    err_test_all = []
    
    for i in range(nfeat):
        trainX_feat = trainX[:,i][:,np.newaxis]
        testX_feat = testX[:,i][:,np.newaxis]
        err_train = predict(trainX_feat,trainY,testX_feat,testY)[2]
        err_test = predict(trainX_feat,trainY,testX_feat,testY)[3]
        err_train_all.append(err_train)
        err_test_all.append(err_test)
        
    return err_train_all, err_test_all