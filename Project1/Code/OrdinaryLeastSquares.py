import numpy as np
from numpy.random import randint
import warnings
import scipy.stats

class OrdinaryLeastSquares:
    ''' Perform OLS regression and assess performance.

    My own code for performing regression using Ordinary Least 
    Squares (OLS)
    Including resampling techniques:
    - K-fold Cross Validation (KFoldCV)
    - Bootstrap
    '''

    def __init__(self):
        
        self.is_regressed = False
        # self.beta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.z)
        # z_prediction = self.X.dot(beta)

    def fit(self,X,z):
        ''' Regression using Ordinary Least Squares.

        X      - The design matrix
        z      - The output vars
        beta   - The parameters beta
        ztilde - The prediction for z
        '''
        # Include this in constructor instead of own method? Assures ztilde 
        # and beta are created when calling other methods

        self.X = X
        self.z = z
        self.n = np.size(z)
        
        self.beta = np.linalg.pinv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.z)
        self.ztilde = self.X.dot(self.beta)
        self.is_regressed = True
       
        return self.ztilde

    def predict(self,input):
        return input.dot(self.beta)

    def bootstrap(self,B,statistic):
        '''Resampling using the bootstrap method

        Calculates an estimate for the statistic given as input,
        on the data z.

        B         - Number of bootstraps
        statistic - the statistic to be estimated (a method)
        returns an array of the calculated theta values
        '''
        theta = zeros(B)
        for i in range(B):
            theta[i] = statistic(z[randint(0,self.n,self.n)]) # Randomly select n vars from z

        return theta


    def mean_square_error(self,z_prediction=None,z_actual=None):
        ''' Returns the Mean Square Error '''
        if z_prediction is None:
            z_prediction = self.ztilde
        if z_actual is None:
            z_actual = self.z

        return np.sum((z_actual-z_prediction)**2)/self.n

    def r2(self,z_prediction=None,z_actual=None):
        '''Returns the R**2 value for z.'''
        if z_prediction is None:
            z_prediction = self.ztilde
        if z_actual is None:
            z_actual = self.z
        
        return 1-np.sum((z_prediction-z_actual)**2)/np.sum((z_prediction-self.mean(z_prediction))**2)

    def mean(self,z):
        ''' Returns the mean of the values of the array z.'''
        return np.sum(z)/self.n

    def variance_of_beta(self):
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




