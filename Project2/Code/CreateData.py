import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numba import jit

class CreateData:
    '''
    A helper class for creating the input and target
    data.

    n       - The number of data points in the x and y 
              arrays
    seed    - Seed for the random number generator
    '''
    def __init__(self,n,seed=9):
        np.random.seed(seed=seed)
        #self.n = n
        x = np.sort(np.random.rand(n))
        y = np.sort(np.random.rand(n))
        self.x_mesh, self.y_mesh = np.meshgrid(x, y)
        #self.x = np.ravel(self.x_mesh)
        #self.y = np.ravel(self.y_mesh)
        self.z_mesh = self.calculate_franke_values(self.x_mesh,self.y_mesh)
        # self.z = np.ravel(self.z_mesh)
        self.split = False # (dataset not (yet) split into train and test)

    def set_values(self,X,z):
        ''' Set Values for the design matrix X and the output z '''
        self.X = X
        self.z = z

    def calculate_franke_values(self,x,y):
        ''' The Franke function. '''
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
        self.z_mesh = self.z_mesh + np.random.normal(mean,variance,self.z_mesh.shape)

    def create_design_matrix(self,d=5,x=None,y=None):
        ''' Set up design matrix X
    
        Builds the design matrix X for a polynomial of degree n
        with two input variables x and y. Uses method outside class 
        for optimalization with numba.jit.
    
        d     - the polynomial degree 
        return the design matrix X
        '''
        if x is None:
            x = np.ravel(self.x_mesh)
        if y is None:
            y = np.ravel(self.y_mesh)
        
        if len(x.shape)>1:
            x = np.ravel(x)
            y = np.ravel(y)

        self.X = create_matrix(x,y,d)
        return self.X

    def plot_data(self,zmin=-0.1,zmax=1.4,X=None,z=None,bar=True,show=True):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot the surface.
        if X is None:
            surf = ax.plot_surface(self.x_mesh, self.y_mesh, self.z_mesh, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        else:
            x = X[:,1]
            y = X[:,2]
            if(z.ndim>1):
                z = np.ravel(z)
            surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(zmin, zmax)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        if bar:
            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)
        if show:
            plt.show()

    def split_dataset(self,test):
        '''
        Splits the created data into three parts;
        training, validation, and test

        test    - fraction of the dataset used for test
        '''
        z = np.ravel(self.z_mesh)
        self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(self.X, z, test_size = test, random_state=3)
        #return train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        self.split = True
        return self.X_train, self.X_test, self.z_train, self.z_test

    def scale_dataset(self,type='standard'):
        '''
        Scales the dataset using scaler from scikit-learn.
        input type gives scaler choice. Possible options include:
        standard:   sklearn.StandardScaler
        minmax:     sklearn.MinMaxScaler
        '''
        self.scaling = type.lower()
        if(self.split):
            if(self.scaling=='standard'):
                # removing the mean and scaling to unit variance
                scaler = StandardScaler()
                scaler.fit(self.X_train[:,1:])                          # Fit to training data
                X_train_scaled = scaler.transform(self.X_train[:,1:])    
                X_test_scaled = scaler.transform(self.X_test[:,1:])     # use same fit when scaling test data
                self.X_test = np.hstack((np.ones((self.X_test.shape[0],1)),X_test_scaled))
                self.X_train = np.hstack((np.ones((self.X_train.shape[0],1)),X_train_scaled))

            elif(self.scaling=='minmax'):
                # Scaling to lie between 0 and 1
                min_max_scaler = MinMaxScaler()
                self.X_train = min_max_scaler.fit_transform(self.X_train) # Fit to training data
                self.X_test = min_max_scaler.transform(self.X_test)       # use same fit when scaling test data

        else:
            if(self.scaling=='standard'):
                scaler = StandardScaler()
                scaler.fit(self.X[:,1:])
                X_scaled = scaler.transform(self.X[:,1:])
                self.X = np.hstack((np.ones((self.X.shape[0],1)),X_scaled))

            elif(self.scaling=='minmax'):
                # Scaling to lie between 0 and 1
                min_max_scaler = MinMaxScaler()
                self.X = min_max_scaler.fit_transform(self.X)

@jit           
def create_matrix(x,y,d):
    n = len(x)
    p = int((d+1)*(d+2)/2)  # number of terms in the resulting polynomial                                                              
    X = np.ones((n,p)) # X has dimensionality nxp

    # Building X:
    for i in range(1,d+1):
        q = int((i)*(i+1)/2)
        for j in range(i+1):
            X[:,q+j] = (x**(i-j))*(y**(j))

    return X
