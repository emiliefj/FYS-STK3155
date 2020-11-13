import numpy as np
import warnings

class LogisticRegression():
    '''
    A class for performing logistic regression for classification using
    stochastic gradient desctent (SGD).

    lmda    - regression parameter l2 regularization

    '''
    def __init__(self, lmda=0, decay=False, seed=687):
        np.random.seed(seed)
        self.lmda = lmda 
        self.decay = decay

    def initialize(self,m,n):
        self.weights = np.random.randn(m, n)/np.sqrt(n)
        #self.biases = np.random.randn(m, 1)

        
    def feed_forward(self):
        z = np.dot(W, x) + b
        a = self.softmax(z)


    def sgd(self, X, y, n_epochs, batchsize, learning_rate, t0, t1, n_classes=10, test_data=None, print_epochs=False):
        '''

        X       - predictors/inputs
        y       - output
        n_epochs        - number of epochs
        batchsize       - size of each batch in stochastic gradient descent
        learning_rate   - the step size or learning rete of the SGD
        t0, t1          - parameters for gradually decreasing step size.
                          only used if self.decay=True
        n_classes       - the number of classes/digits to in the dataset
        
        '''
        # initialize
        n = len(X)
        n_batches = int(len(X)/batchsize)
        if(test_data):
            X_test, y_test = zip(*test_data)
            n_test = len(X_test)
        if(y.ndim==1): # only for classification of 10 things. Make optional, and more general (i.e. varying number of outputs)
           y = np.array([self._vector_transform(yi,n_classes) for yi in y])
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

    def softmax(self, z):
        '''
        numerically stable softmax function avoiding overflow issues
        [8]
        '''
        return np.exp(z-max(z)) / np.sum(np.exp(z-max(z)))

    def _vector_transform(self,i,k):
        ''''
        Transform input i to a 10d vector with 1. on the ith index (the
        index corresponding to the number-value.
        '''
        vector = np.zeros((k,))
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