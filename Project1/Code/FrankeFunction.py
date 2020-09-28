from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


class FrankeFunction:
    '''
    The Franke function is a two dimensional weighted sum of four 
    exponentials. It has two Gaussian peaks of different heights, 
    and a smaller dip and is often used as a test function in 
    interpolation problems.

    x     --- the x values
    y     --- the y values

    Ref: Franke, R. (1979). A critical comparison of some methods 
    for interpolation of scattered data
    '''
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.z = self.calculate_values(x,y)

    def calculate_values(self,x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4
    
    def get_values(self):
        return self.z

    def add_normal_noise(self,mean,variance):
        '''
        Adds noise to the data in the form of random
        numbers following a normal distribution with
        mean and variance as given in the input.

        mean        -- the mean of the distribution
        variance     -- sigma**2 of the distribution
        '''
        self.z = self.z + np.random.normal(mean,variance,self.z.shape)

    def plot_function(self):
        '''
        Creates a meshgrid of the x and y values, 
        calculates the corresponding z-values,
        and plots the result as a surface plot.
        '''
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Plot the surface.
        x_grid, y_grid = np.meshgrid(self.x,self.y)
        z_grid = self.calculate_values(x_grid,y_grid)
        surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

