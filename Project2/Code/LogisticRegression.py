import numpy as np
import warnings

class LogisticRegression():
    '''
    A class for performing logistic regression for classification using
    stochastic gradient desctent (SGD).

    lmda    - regression parameter l2 regularization

    '''
    def __init__(self,lmda=0,seed=687):
        np.random.seed(seed)
        self.lmda = lmda 

    def sgd(self, X, y, n_epochs, batchsize, learning_rate, t0, t1, test_data=None, print_epochs=False):
        n = len(X)
        n_batches = int(len(X)/batchsize)
        if(test_data):
            X_test, y_test = zip(*test_data)
            n_test = len(X_test)
        #if(y.ndim==1): # only for classification of 10 things. Make optional, and more general (i.e. varying number of outputs)
        #    y = np.array([self._vector_transform(yi) for yi in y])
        #beta_0 = np.random.randn(len(self.X[0]),1) # Make initial guess for beta
        for i in range(0,self.n_epochs):
            for b in range(0,n_batches):
                # fetch X and y of current mini-batch

                batch = np.random.randint(n,size=batchsize)
                current_X = X[batch]
                current_z = z[batch].reshape(batchsize,1)
                # calculate gradient
                gradient = (current_X.T.dot(current_X.dot(beta_0)-current_z))*2/batchsize-self.alpha*(beta_0)
                beta_0 = beta_0-self.learning_schedule(t0,t1,i*n_batches+b)*gradient 
        return beta_0

    def learning_schedule(self, t0, t1, t):
        if self.decay:
            return self.t0/(t+self.t1)
        else:
            return self.learning_rate

    def _vector_transform(self,i):
        ''''
        Transform input i to a 10d vector with 1. on the ith index (the
        index corresponding to the number-value.
        '''
        vector = np.zeros((10,))
        vector[i] = 1.0
        return vector

def accuracy(pred, actual):
    ''' 
    A function for measuring the accuracy of classification
    Returns an accuracy score as the fraction of predictions that are 
    correct.

    Accuracy = sum(correct predictions)/number of prediction

    pred    - the prediction made by the model
    actual  - the actual value in the data 
    '''
    n = len(pred)
    correctly_predicted = sum(int(y == t) for (y, t) in zip(pred,actual))

    return correctly_predicted/n