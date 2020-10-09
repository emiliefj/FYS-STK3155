import numpy as np
from numpy.random import randint
import warnings
import scipy.stats
from sklearn.utils import resample
import random

import Code.OrdinaryLeastSquares as OLS

class RidgeRegression(OLS.OrdinaryLeastSquares):

    def fit(self,X,z,alpha=0.1):
        ''' Regression using Ordinary Least Squares.

        X       - The design matrix
        z       - The output vars
        alpha   - parameter for regularization strength,
                  must be positive
        ztilde  - The prediction for z
        '''
        self.X = X
        self.z = z
        self.n = np.size(z)
        self.alpha = alpha
        p = self.X.shape[1]
        self.beta = np.linalg.pinv(self.X.T.dot(self.X)+self.alpha*np.eye(p,p)).dot(self.X.T).dot(self.z)
        self.ztilde = self.X.dot(self.beta)
        self.is_regressed = True
       
        return self.ztilde

    def k_fold_cv(self,input_data,target,k=5, alpha=0.1, shuffle=True):
        ''' resampling using k-fold cross validation

        input_data  - input data
        target      - target data
        k           - The number of folds
        shuffle     - if True the dataset is shuffled before 
                      splitting into folds
        returns the average of the mean square errors found when predicting the test fold
        '''

        indices = np.array(range(0,input_data.shape[0]))
        if(shuffle): # shuffle data before split
            indices = random.sample(range(0,input_data.shape[0]), input_data.shape[0])

        folds = np.array_split(indices, k)

        mse = np.zeros(k)
        r2 = np.zeros(k)
        for i in range(k):
            train = folds.copy()
            test = train[i]
            del train[i]
            train = np.concatenate(train)
            self.fit(input_data[train],target[train],alpha)
            target_hat = self.predict(input_data[test])
            mse[i] = self.mean_square_error(target_hat,target[test])
            r2[i] = self.r2(target_hat,target[test])

        return np.mean(mse), np.mean(r2)

    def bootstrap_fit(self,train_data,train_target,test_data,test_target,B=100,alpha=0.1): 
        ''' Resampling using the bootstrap method

        train_data      - input data used for fit
        train_target    - target data used for fit
        test_data       - input data used for test
        test_target     - target data used for test
        B               - The number of bootstraps

        returns the average mse when predicting the test set in each of 
        the B bootstraps
        '''
        mse_train = np.zeros((B))
        mse_test = np.zeros((B))
        for b in range(B):
            X_, z_ = resample(train_data, train_target)       # shuffle data, with replacement
            self.fit(X_,z_,alpha=alpha) # fit model
            # prediction on same test data every time
            mse_train[b] = self.mean_square_error(self.predict(X_),z_)
            mse_test[b] = self.mean_square_error(self.predict(test_data),test_target)

        return np.mean(mse_test),np.mean(mse_train)

    def variance_of_beta(self):
        '''Finds the variance for the parameters beta.'''
        n, p = self.X.shape # dimensions
        if((n-p)==1):
            n = n+1 # avoid dividing by zero
        var_hat = 1./(n-p-1)*np.sum((self.z-self.ztilde)**2)
        xtx_alphaI = np.linalg.pinv(self.X.T.dot(self.X)-self.alpha*np.eye(p,p))
        return np.diagonal(var_hat*xtx_alphaI.dot(self.X.T).dot(self.X).dot((xtx_alphaI).T))