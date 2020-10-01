import numpy as np
from numpy.random import randint
import warnings
import scipy.stats
from sklearn.utils import resample

# import CreateData as cd

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
        ''' Finding a better estimate for a statistic
        theta using the bootstrap method

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

    def bootstrap_fit(self,data_creator,B,max_degree,n=100):
        ''' Resampling using the bootstrap method

        data_creator- class for creating data
        B           - Number of bootstraps
        max_degree  - max polynomial degrees
        n           - number of datapoints in training set for
                      the models
        returns an array of dimensions (max_degree,B) with 
        z-predictions for each model variation on the same test set
        '''
        test_data = data_creator.CreateData(n) # Create test data
        z_test = test_data.z

        data = data_creator.CreateData(n)              # Create data for model fit
        mse_test = np.zeros((max_degree,B))            # For storing results
        mse_train = np.zeros((max_degree,B)) 
        avg_mse_train = np.zeros(max_degree)
        avg_mse_test = np.zeros(max_degree)
        print("--- Mean square error on static test set for B=%d bootstraps. ---" %(B))
        for d in range(max_degree):
            test_data.create_design_matrix(d+1)         # Building design matrix with test data
            test_data.scale_dataset() # Not scaled the same as test set - uh oh
            data.create_design_matrix(d+1)              # Building design matrix for model fit
            data.scale_dataset()

            mse_train
            for i in range(B):
                X_, z_ = resample(data.X, data.z)       # shuffle data, with replacement
                self.fit(X_,z_)                         # fit to shuffled data

                # prediction on same test data every time
                #print(z_test-self.predict(test_data.X))
                mse_train[d,i] = self.mean_square_error(self.predict(X_),z_)
                mse_test[d,i] = self.mean_square_error(self.predict(test_data.X),z_test)
            avg_mse_test[d] = np.mean(mse_test[d,:])
            avg_mse_train[d] = np.mean(mse_train[d,:])
            print(f"d=%d: MSE(test set): %f " %(d+1,avg_mse_test[d]))

        return avg_mse_test, avg_mse_train

    def k_fold_cv(self):
        pass
    
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




