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
        
        p = self.X.shape[1]
        self.beta = np.linalg.pinv(self.X.T.dot(self.X)+alpha*np.eye(p,p)).dot(self.X.T).dot(self.z)
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
        for i in range(k):
            train = folds.copy()
            test = train[i]
            del train[i]
            train = np.concatenate(train)
            self.fit(input_data[train],target[train],alpha)
            target_hat = self.predict(input_data[test])
            mse[i] = self.mean_square_error(target_hat,target[test])

        return np.mean(mse)