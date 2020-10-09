import numpy as np
#from sklearn.metrics import mean_squared_error, r2_score

class ErrorMeasure:
    ''' Class holding methods for several error measures.'''

    def MSE(self,z,ztilde):
        ''' Returns the Mean Square Error of z.'''
        return np.sum((z-ztilde)**2)/np.size(z)

    def R2(self,z,ztilde):
        '''Returns the R**2 value for z.'''
        return 1-np.sum((z-ztilde)**2)/np.sum((z-self.mean(z))**2)

    def mean(self,z):
        ''' Returns the mean of the values of the array z.'''
        return np.sum(z)/np.size(z)

# y = np.arange(0,1,0.01)
# ytilde = y+np.random.normal(0,0.001,np.size(y))
# em = ErrorMeasure()
# mse = em.MSE(y,ytilde)
# sklearn_mse = mean_squared_error(y,ytilde)
# print(mse)
# print(sklearn_mse)

# r2 = em.R2(y,ytilde)
# sklearn_r2 = r2_score(y,ytilde)
# print(str(r2) + "   "+str(sklearn_r2))
# a = np.arange(0, 4, 1)
# b = np.arange(1,5,1)
# c = (a+b)
# print(a)