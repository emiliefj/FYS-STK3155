import numpy as np
import warnings
import scipy.stats
from sklearn.utils import resample
import random
import pandas as pd

class LinearRegressionwithGradientDescent:
    ''' Perform linear regression and assess performance.

    My own code for performing regression using Ordinary Least 
    Squares (OLS) with gradient descent
    Including resampling techniques:
    - K-fold Cross Validation (KFoldCV)
    - Bootstrap
     setting alpha adds L2 regularization
    '''

    def __init__(self,seed=199, method="ols", alpha=0, n_epochs=10, batchsize=1, learning_rate=0.01, max_iter=1000, decay=False, t0=1.0, t1=10):
        random.seed(seed)
        self.method = method.lower()
        self.alpha = alpha # aka lambda 
        self.n_epochs = n_epochs
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.decay = decay
        self.t0 = t0
        self.t1 = t1

        self.is_regressed = False

    def fit(self,X,z):
        ''' Regression using Ordinary Least Squares.
        Finds beta, the parameters or weights of the fit and finds
        the prediction for z, ztilde

        X      - The design matrix
        z      - The output vars
        returns ztilde, the prediction for z.
        '''

        self.X = X
        self.z = z
        self.n = len(X)

        #self.n = len(X)
        #self.p = len(X[0])
        self.beta = self.findBetas()
        self.ztilde = self.X.dot(self.beta)
        
        self.is_regressed = True
       
        return self.ztilde

    def findBetas(self):
        if self.method=="ols":
            return np.linalg.pinv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.z)

        if self.method=="sgd":
            return self.sgd()


    def sgd(self):
        n_batches = int(len(self.X)/self.batchsize)
        beta_0 = np.random.randn(len(self.X[0]),1) # Make initial guess for beta
        for i in range(0,self.n_epochs):
            for b in range(0,n_batches):
                # fetch X and y of current mini-batch

                batch = np.random.randint(self.n,size=self.batchsize)
                current_X = self.X[batch]
                current_z = self.z[batch].reshape(self.batchsize,1)
                # calculate gradient
                gradient = (current_X.T.dot(current_X.dot(beta_0)-current_z))*2/self.batchsize-self.alpha*(beta_0)
                beta_0 = beta_0-self.learning_schedule(i*n_batches+b)*gradient 
        return beta_0

    def learning_schedule(self,t):
        if self.decay:
            return self.t0/(t+self.t1)
        else:
            return self.learning_rate

    def predict(self,input_data):
        ''' Making a prediction for the output using the regression model found in fit() '''
        if not self.is_regressed:
            warnings.warn('Call .fit()-method first to create model with training data.')
            return
        return input_data.dot(self.beta)

    def bootstrap(self,B,statistic):
        ''' Finding a better estimate for a statistic
        theta using the bootstrap method.

        Calculates an estimate for the statistic given as input,
        on the data z.

        B         - Number of bootstraps
        statistic - the statistic to be estimated (a method)
        returns an array of the calculated theta values
        '''
        rng = np.random.default_rng()
        theta = zeros(B)
        for i in range(B):
            theta[i] = statistic(z[rng.integers(0,self.n,self.n)])  # Randomly select n vars from z
                                                                    # with replacement

        return theta

    def k_fold_cv(self,input_data,target,k=5, shuffle=True):
        ''' Resampling using k-fold cross validation

        input_data  - input data
        target      - target data
        k           - The number of folds
        shuffle     - if True the dataset is shuffled before 
                      splitting into folds
        returns the average of the mean square errors found when predicting the test fold
        '''
        n = input_data.shape[0]
        indices = np.array(range(0,n))
        if(shuffle): # shuffle data before split
            indices = random.sample(range(0,n), n)

        folds = np.array_split(indices, k)

        mse = np.zeros(k)
        r2 = np.zeros(k)
        for i in range(k):
            train = folds.copy()
            test = train[i]
            del train[i]
            train = np.concatenate(train)
            self.fit(input_data[train],target[train])
            target_hat = self.predict(input_data[test])
            mse[i] = self.mean_square_error(target_hat,target[test])
            r2[i] = self.r2(target_hat,target[test])

        return np.mean(mse), np.mean(r2)

    def bootstrap_fit(self,train_data,train_target,test_data,test_target,B=100): 
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
            self.fit(X_,z_) # fit model
            # prediction on same test data every time
            mse_train[b] = self.mean_square_error(self.predict(X_),z_)
            mse_test[b] = self.mean_square_error(self.predict(test_data),test_target)

        return np.mean(mse_test),np.mean(mse_train)

    def mean_square_error(self,z_prediction=None,z_actual=None):
        ''' Returns the Mean Square Error '''
        if z_prediction is None:
            z_prediction = self.ztilde
        if z_actual is None:
            z_actual = self.z
        if(z_prediction.shape!=z_actual.shape):
            z_prediction = z_prediction.reshape((-1, 1))
            z_actual = z_actual.reshape((-1, 1))
        
        return np.mean((z_actual-z_prediction)**2)

    def r2(self,z_prediction=None,z_actual=None):
        '''Returns the r^2 value for z.'''
        if z_prediction is None:
            z_prediction = self.ztilde
        if z_actual is None:
            z_actual = self.z
        
        return 1-np.sum((z_actual-z_prediction)**2)/np.sum((z_actual-self.mean(z_actual))**2)

    def mean(self,z):
        ''' Returns the mean of the values of the array z.'''
        return np.sum(z)/np.size(z)

    def variance_of_beta(self): # Make private? __variance_of_beta(self)
        '''Finds the variance for the parameters beta.'''
        n, p = self.X.shape # dimensions
        if((n-p)==1):
            n = n+1 # avoid dividing by zero
        var_hat = 1./(n-p-1)*np.sum((self.z-self.ztilde)**2)
        return np.diagonal(var_hat*np.linalg.pinv(self.X.T.dot(self.X)))

    def sigma2(self,z):
        '''Finds the variance of z.''' 
        return np.sum((z-np.mean(z))**2)/(np.size(z)-1)

    def get_beta_CIs(self,confidence):
        if not self.is_regressed:
            warnings.warn("Call model.fit(X,z) first to create beta values")
            return

        if(self.n<40):
            warnings.warn("The expression used for calculating the confidence interval of beta is valid for large samples. You may get inaccurate results.")

        if(confidence>1): # confidence given as a percentage, not a fraction
            confidence = confidence/100.
        
        std_beta = np.sqrt(self.variance_of_beta())
        bound = scipy.stats.t.ppf((1+confidence)/2.,np.size(self.beta)-1)*std_beta
        
        # I dont have mean(beta_i), using found value beta_i under the assumption it is the expectation value of beta
        lower = self.beta-bound
        upper = self.beta+bound
        return np.vstack([lower,upper]).T




