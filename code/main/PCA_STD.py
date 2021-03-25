#!usr/bin/python3.6
  
"""
Created on Sat Jan 2 2021

@author: Pegah Mirabedini
"""

"""
you need to define the train and test datasets, and preprocess them if necessary.
trainX: train input: shape of (number of samples, number of features)
trainY: train target: shape of (number of samples,)
"""

#import libraries:
import numpy as np
import matplotlib.pyplot as plt

def PCA_STD(trainX,testX,n_component):
    """
    PCA to transform features using standard deviation: keeping the first n_components
    output: The transformed train and test sets.
    """
    #calculate the emperical mean of train data
    STD = np.std(trainX, axis=0)
    STD[STD==0] += 1
    norm_trainX = (trainX - np.mean(trainX,axis=0))/STD
    norm_testX = (testX - np.mean(testX,axis=0))/STD
    #calculate U,V,D (U:eigenvalues of (X-Xavg): (1,m); V:eigenvectors of (X-Xavg)(m,m); D: Covarience matrix: XTX
    D = np.cov((norm_trainX).T)
    U , V = np.linalg.eigh(D) 
    # sort the columns of V and U according to D (from highest to lowest eigenvalue) and keep the first n_components
    indx_ranked = U.argsort()[::-1][:n_component]
    U_ranked = U[indx_ranked]
    V_ranked = V[:,indx_ranked]
    # find reconstructed X: Xnew
    X_train_new = np.dot(V_ranked.T, (norm_trainX).T).T
    X_test_new = np.dot(V_ranked.T, (norm_testX).T).T
    return X_train_new, X_test_new