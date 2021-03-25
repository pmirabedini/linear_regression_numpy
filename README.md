## Linear Regression implemented in NumPy
#### The train and test dataset should to be defined by the user.

 This code provides linear regression and principal components Analysis (PCA) functions and postprocessing to analyze feature selection. This code was written as a class project. Therefore, I am only providing the parts I wrote. You can simply use python plot options to visualize the results from each part. 


**○ How does it work?**

#### preprocessing of data
The preprocessing code is not included in this repository. For reruning the code, you need to provide the data in the following format:

- *trainX* and *trainY*: the training data, *trainX* (number of samples, number of features) *trainY* (number of samples,).  
- *testX* and *testY*: the testing data, similar shape as the train data (with different number of samples)
- *featurenames*: an array of the features names
- *nfeat*: number of the features


#### Main functions and visualization - my contribution
- part one: linear regression model. Trains on the training set and calculates the mean squared error on both the training set and the testing set. You can use the postprocess function get_lr_errors.py to analyze the  training error and the testing error when using each feature individually.  You can also use the get_error_addfeat.py to add the features in the order they predict y individually from best to worst. 

- part two: function to perform PCA transformation on the data

- part three: PCA transformation scaled each feature independently by its standard deviation (z-normalization).

Similar to part one, you can use the get_PCA_errors.py function to find the train and test errors when using each feature individually. 


 




