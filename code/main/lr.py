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


def mean_sqrr_err(Y,Ypred):
    """
    function to compute mean squared error.
    Y, Ypred: 1D array
    """
    #using broadcasting to add to the dimensionality of y.
    m = Y.shape[0]
    Yext = Y[:,np.newaxis]
    Ypred_ext = Ypred[:,np.newaxis]
    squared_loss = ((Ypred_ext-Yext)**2).sum()
    
    return squared_loss/m


def add_ones(X):
    """
    function to add a row of ones (regularization) to X.
    X: matrix(mxn)
    """
    m = X.shape[0]
    ones = np.ones((m,1))
    Xn = np.hstack((ones,X))
    
    return Xn

    
def learn(X,Y):
    """
    function to train model.
    X: matrix(mxn)
    Y: 1D array
    """
    Xn = add_ones(X)
    Yext = Y[:,np.newaxis]
    #calculates (XTX)^-1
    pinv_X = np.linalg.pinv(Xn) 
    W = pinv_X@Y
    
    return W


def predict(trainX,trainY,testX,testY):
    """
    function to predict y and compute error for train and test sets.
    trainX, testX: matrix(mxn)
    trainY, testY: 1D array
    """
    W = learn(trainX,trainY)
    Ypred_test = add_ones(testX)@W 
    Ypred_train = add_ones(trainX)@W
    err_test = mean_sqrr_err(testY,Ypred_test)
    err_train = mean_sqrr_err(trainY,Ypred_train)
    
    return Ypred_train, Ypred_test, err_train, err_test


def predict_to_converge(trainX,trainY,testX,testY,err_max=1000):
    """
    function to predict y and compute error for train and test sets until convergence to a threshold. 
    trainX, testX: matrix(mxn)
    trainY, testY: 1D array
    """
    err_test = 10000
    
    while err_test > err_max:
        W = learn(trainX,trainY)
        Ypred_test = add_ones(testX)@W 
        Ypred_train = add_ones(trainX)@W
        err_test = mean_sqrr_err(testY,Ypred_test)
        err_train = mean_sqrr_err(trainY,Ypred_train)
    
    return Ypred_train, Ypred_test, err_train, err_test
