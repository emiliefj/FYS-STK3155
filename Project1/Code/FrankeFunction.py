from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
#from random import random, seed
#from numba import jit

class FrankeFunction:
    '''
    The franke function is a two dimensional weighted sum of four 
    exponentials. It has two Gaussian peaks of different heights, 
    and a smaller dip and is often used as a test function in 
    interpolation problems.

    x     --- the x values
    y     --- the y values

    Ref: Franke, R. (1979). A critical comparison of some methods 
    for interpolation of scattered data
    '''
    def __init__(self,x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        self.x = x
        self.y = y
        self.z = term1 + term2 + term3 + term4
    
    def get_values(self):
        return self.z

    def add_normal_noise(self,mean,variance):
        '''
        Adds some noise to the data in the form of
        random numbers following a normal distribution
        with mean=0 and variance as given in the input.

        mean        -- the mean of the distribution
        variance     -- sigma**2 of the distribution
        '''
        self.z = self.z + np.random.normal(mean,variance,self.z.shape)

    def plot_function(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_surface(self.x, self.y, self.z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

# Make data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
franke_function = FrankeFunction(x, y)
franke_function.add_normal_noise(0,0.01)
franke_function.plot_function()


# ----------

# @jit
def create_design_matrix(x,y,d): # Not sure where to put this yet
    ''' Set up design matrix X

    Builds the design matrix X for a polynomial of degree n
    with two input variables x and y.

    x     - first input variable (array of length N)
    y     - second input variable (array of length N)
    d     - the polynomial degree 
    return the design matrix X
    '''
    n = len(x)
    if(n!=len(y)):
        raise Exception("Input args x and y differ in length")

    p = int((d+1)*(d+2)/2)  # number of terms in the resulting polynomial                                                              
    X = np.ones((n,p))      # X has dimensionality nxp

    # Building X:
    for i in range(1,d+1):
        q = int((i)*(i+1)/2)
        for j in range(i+1):
            X[:,q+j] = (x**(i-j))*(y**(j))

    return X