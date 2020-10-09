import numpy as np
import random
from sklearn.linear_model import Lasso
import Code.RidgeRegression as RR

class LassoRegression(RR.RidgeRegression):
    
    def __init__(self,alpha=1.0,seed=199,max_iter=1000):
        random.seed(seed)     
        self.is_regressed = False
        self.model = Lasso(alpha=alpha,max_iter=max_iter)
        

    def fit(self,X,z,alpha=None):
        self.X = X
        self.z = z
        self.n = np.size(z)

        if alpha is not None:
            self.model = self.model.set_params(alpha=alpha)
        
        z_hat = self.model.fit(X,z)
        self.is_regressed = True
        self.beta = self.model.coef_
        
        return z_hat

    def predict(self,input_data):
        ''' Making a prediction for the output using the regression model found in fit() '''
        if not self.is_regressed:
            warnings.warn('Call .fit()-method first to create model with training data.')
            return
        return input_data.dot(self.beta)



