import numpy as np
import warnings
import scipy.stats
from sklearn.utils import resample
import random

class OrdinaryLeastSquares:
    ''' Perform OLS regression and assess performance.

    My own code for performing regression using Ordinary Least 
    Squares (OLS)
    Including resampling techniques:
    - K-fold Cross Validation (KFoldCV)
    - Bootstrap
    '''

    def __init__(self,seed=199):
        random.seed(seed)     
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
        self.n = np.size(z)
        
        self.beta = np.linalg.pinv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.z)
        self.ztilde = self.X.dot(self.beta)
        
        self.is_regressed = True
       
        return self.ztilde

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
        for i in range(k):
            train = folds.copy()
            test = train[i]
            del train[i]
            train = np.concatenate(train)
            self.fit(input_data[train],target[train])
            target_hat = self.predict(input_data[test])
            mse[i] = self.mean_square_error(target_hat,target[test])

        return np.mean(mse)

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

    def bootstrap_fit_old(self,input_data,target,B=100,test_fraction=0.25):
        ''' Resampling using the bootstrap method

        input_data  - input data
        target      - target data
        B           - The number of bootstraps
        test_fraction   - the fraction of the data used for test

        returns the average mse when predicting the test set in each of 
        the B bootstraps
        '''
        # Leaving a fraction of the data for test
        test_indices = random.sample(range(0,input_data.shape[0]), int(input_data.shape[0]*test_fraction))
        input_test = input_data[test_indices]
        input_data = np.array(np.delete(input_data,test_indices,0))
        target_test = target[test_indices]
        target = np.array(np.delete(target,test_indices,0)) #np.setdiff1d(target,target_test)

        mse_train = np.zeros((B))
        mse_test = np.zeros((B))
        for b in range(B):
            #indices = random.choices(range(0,input_data.shape[0]), input_data.shape[0])
            X_, z_ = resample(input_data, target,random_state=13)       # shuffle data, with replacement
            self.fit(X_,z_) # fit model
            # prediction on same test data every time
            mse_train[b] = self.mean_square_error(self.predict(X_),z_)
            mse_test[b] = self.mean_square_error(self.predict(input_test),target_test)

        return np.mean(mse_test),np.mean(mse_train)

    def bootstrap_fit_loop(self,data_creator,B=100,max_degree=10,n=100): # maybe move loop over d out?
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

        # Create data for model fit
        data = data_creator.CreateData(n)
        # For storing results
        mse_test = np.zeros((max_degree,B))
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
                X_, z_ = resample(data.X, data.z, random_state=13)       # shuffle data, with replacement
                self.fit(X_,z_)                         # fit to shuffled data

                # prediction on same test data every time
                mse_train[d,i] = self.mean_square_error(self.predict(X_),z_)
                mse_test[d,i] = self.mean_square_error(self.predict(test_data.X),z_test)
            avg_mse_test[d] = np.mean(mse_test[d,:])
            avg_mse_train[d] = np.mean(mse_train[d,:])
            print(f"d=%d: MSE(test set): %f " %(d+1,avg_mse_test[d]))

        return avg_mse_test, avg_mse_train

    def mean_square_error(self,z_prediction=None,z_actual=None):
        ''' Returns the Mean Square Error '''
        if z_prediction is None:
            z_prediction = self.ztilde
        if z_actual is None:
            z_actual = self.z

        return np.sum((z_actual-z_prediction)**2)/self.n

    def r2(self,z_prediction=None,z_actual=None):
        '''Returns the r^2 value for z.'''
        if z_prediction is None:
            z_prediction = self.ztilde
        if z_actual is None:
            z_actual = self.z
        
        return 1-np.sum((z_prediction-z_actual)**2)/np.sum((z_prediction-self.mean(z_prediction))**2)

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




