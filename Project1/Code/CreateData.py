import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class CreateData:
    '''
    A helper class for creating the input and target
    data.

    n   - The number of data points in the x and y 
          arrays
    '''
    def __init__(self,n,seed):
        np.random.seed(seed=9)
        self.n = n
        self.x = np.random.rand(n)#np.linspace(0,1,self.n)
        self.y = np.random.rand(n)#np.linspace(0,1,self.n)
        self.z = self.calculate_values(self.x,self.y)

    def calculate_values(self,x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def add_normal_noise(self,mean,variance):
        '''
        Adds noise to the data in the form of random
        numbers following a normal distribution with
        mean and variance as given in the input.

        mean        -- the mean of the distribution
        variance     -- sigma**2 of the distribution
        '''
        self.z = self.z + np.random.normal(mean,variance,self.z.shape)

    def create_design_matrix(self,d):
        ''' Set up design matrix X
    
        Builds the design matrix X for a polynomial of degree n
        with two input variables x and y.
    
        x     - first input variable (array of length N)
        y     - second input variable (array of length N)
        d     - the polynomial degree 
        return the design matrix X
        '''

    
        p = int((d+1)*(d+2)/2)  # number of terms in the resulting polynomial                                                              
        self.X = np.ones((self.n,p))      # X has dimensionality nxp

        # Building X:
        for i in range(1,d+1):
            q = int((i)*(i+1)/2)
            for j in range(i+1):
                self.X[:,q+j] = (self.x**(i-j))*(self.y**(j))

        return self.X

    def split_dataset(self,test):
        '''
        Splits the created data into three parts;
        training, validation, and test

        test    - fraction of the dataset used for test
        '''
        self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(self.X, self.y, test_size = test, random_state=3)
        #return train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        return self.X_train, self.X_test, self.z_train, self.z_test

    def scale_dataset(self,type='standard'):
        '''
        Scales the dataset using scaler from scikit-learn.
        input type gives scaler choice. Possible options include:
        standard:   sklearn.StandardScaler
        minmax:     sklearn.MinMaxScaler
        '''
        self.scaling = type.lower()
        if(self.scaling=='standard'):
            # removing the mean and scaling to unit variance
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            self.X_train_scaled = scaler.transform(self.X_train)    # Fit to training data
            self.X_test_scaled = scaler.transform(self.X_test)      # use same fit when scaling test data

        elif(self.scaling=='minmax'):
            # Scaling to lie between 0 and 1
            min_max_scaler = MinMaxScaler()
            self.X_train_scaled = min_max_scaler.fit_transform(self.X_train) # Fit to training data
            self.X_test_scaled = min_max_scaler.transform(self.X_test)       # use same fit when scaling test data


