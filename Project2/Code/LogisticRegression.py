import numpy as np
import warnings

class LogisticRegression():
    '''
    A class for performing logistic regression for classification using
    stochastic gradient descent (SGD). The cost function used is cross-
    entropy.

    lmda    - regression parameter l2 regularization
    decay   - if True the learning rate decays as we progress through
              the epochs
    seed    - seed for numpy.random for reproduceability
    '''
    def __init__(self, lmda=0, decay=False, seed=687):
        np.random.seed(seed)
        self.lmda = lmda 
        self.decay = decay


    def sgd(self, X, y, n_epochs, batchsize, learning_rate, t0=1, t1=10, n_classes=10, print_epochs=False):
        '''
        Trains the logistic regression model according to the input 
        data using stochastic gradient descent to gradually move 
        down the gradient of the cost function.

        X       - predictors/inputs
        y       - output
        n_epochs        - number of epochs
        batchsize       - size of each batch in stochastic gradient descent
        learning_rate   - the step size or learning rete of the SGD
        t0, t1          - parameters for gradually decreasing step size.
                          only used if self.decay=True
        n_classes       - the number of classes/digits to in the dataset
        print_epochs    - if True a line is printed at the end of each epoch
                          to keep track of progress

        '''
        # initialize
        n, p = X.shape
        self.betas = np.random.randn(p,n_classes)/np.sqrt(n)#np.zeros([p,n_classes])
        n_batches = int(n/batchsize)
        if(y.ndim==1): # transform each yi to a vector with length n_classes
           y = np.array([self._vector_transform(yi,n_classes) for yi in y])

        for i in range(0,n_epochs):
            for b in range(0,n_batches):
                # fetch X and y of current mini-batch
                batch = np.random.randint(n,size=batchsize)
                current_X = X[batch]
                current_y = y[batch] 
                # calculate gradient
                gradient = self.gradient(current_X,current_y,batchsize)
                # update beta
                self.betas = (1-learning_rate*self.lmda)*self.betas-learning_rate*gradient
            if print_epochs:
                print("Epoch {} complete".format(i+1))

        return self.betas

    def learning_schedule(self, t0, t1, t):
        if self.decay:
            return self.t0/(t+self.t1)
        else:
            return self.learning_rate

    def gradient(self, X, y, n):
        '''
        calculate gradient dC/d\beta on the inputted batch X,y
        '''
        prob = self.softmax(np.dot(X, self.betas))
        return  (-1/n)*np.dot(X.T,(y-prob))

    def predict(self, X):
        '''
        Use the trained model to predict the output/classification
        from the input X.
        '''
        prob = self.softmax(np.dot(X, self.betas))

    def softmax(self, z):
        '''
        softmax function for finding the propabilities and making
        predictions.
        '''
        #exponent = np.exp(z-np.max(z))
        exponent = np.exp(z)
        return  exponent/np.sum(exponent, axis=1, keepdims=True)

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
    correctly_predicted = 0
    for i in range(n):
        if(pred[i]==actual[i]):
            correctly_predicted += 1


    return correctly_predicted/n