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

def PCA_errors(PCA_func,trainX,testX):

	""" runs PCA for earch feature and returns the train error and the test errors.
	You need to run PCA.py for your data first.
	"""
    all_test_errors = []
    all_train_errors = []
    
    for k in range(1,nfeat):
        X_train_new, X_test_new = PCA_func(trainX,testX,k)
        _, _, train_err_new, test_err_new = predict(X_train_new,trainY,X_test_new,testY)
        all_train_errors.append(train_err_new)
        all_test_errors.append(test_err_new)
    return all_train_errors, all_test_errors