import numpy as np
from numpy.random import randint
import warnings
import scipy.stats
from sklearn.utils import resample
import random

import Code.OrdinaryLeastSquares as OLS

class RidgeRegression(OLS.OrdinaryLeastSquares):

    def fit(self,X,z,alpha=1.0):
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