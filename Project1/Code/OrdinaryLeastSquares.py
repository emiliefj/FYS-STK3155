import numpy as np
from numpy.random import randint

class OrdinaryLeastSquares:
    ''' Perform OLS regression and assess performance.

    My own code for performing regression using Ordinary Least 
    Squares (OLS)
    Including resampling techniques:
    - K-fold Cross Validation (KFoldCV)
    - Bootstrap
    '''

    def __init__(self,X,z):
        self.X = X
        self.z = z
        self.n = np.size(z)
        # self.beta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.z)
        # self.ztilde = self.X.dot(beta)

    def regress(self):
        ''' Regression using Ordinary Least Squares.

        X      - The design matrix
        z      - The output vars
        beta   - The parameters beta
        ztilde - The prediction for z
        '''
        # Include this in constructor instead of own method? Assures ztilde 
        # and beta are created when calling other methods
        self.beta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.z)
        self.ztilde = self.X.dot(beta)
        return ztilde

    def bootstrap(self,B,statistic):
        '''Resampling using the bootstrap method

        Calculates an estimate for the statistic given as input,
        on the data z.

        B         - Number of bootstraps
        statistic - the statistic to be estimated
        returns an array of the calculated theta values
        '''
        theta = zeros(B)
        for i in range(B):
            theta[i] = statistic(z[randint(0,self.n,self.n)]) # Randomly select n vars from z


    def mean_square_error(self):
        ''' Returns the Mean Square Error of z.'''
        return np.sum((self.z-self.ztilde)**2)/self.n

    def R2(self):
        '''Returns the R**2 value for z.'''
        return 1-np.sum((self.z-self.ztilde)**2)/np.sum((self.z-self.mean(self.z))**2)

    def mean(self):
        ''' Returns the mean of the values of the array z.'''
        return np.sum(self.z)/self.n

    def variance_of_beta(self):
        self.var_of_beta = self.sigma2*np.linalg.inv(self.X.T.dot(self.X))

    def sigma2(self):
        '''Finds the variance of z.'''
        # How?



