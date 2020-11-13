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


    def sgd(self, X, y, n_epochs, batchsize, learning_rate, t0=1, t1=10, n_classes=10, test_data=None, print_epochs=False):
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
        n, p = X.shape
        self.betas = np.zeros([p,n_classes])#np.random.randn(p,n_classes)/np.sqrt(n)#np.zeros([p,n_classes]) #
        n_batches = int(n/batchsize)
        if(test_data):
            X_test, y_test = zip(*test_data)
            n_test = len(X_test)
        if(y.ndim==1): # transform each yi to a vector with length n_classes
           y = np.array([self._vector_transform(yi,n_classes) for yi in y])

        for i in range(0,n_epochs):
            for b in range(0,n_batches):
                # fetch X and y of current mini-batch
                batch = np.random.randint(n,size=batchsize)
                current_X = X[batch]
                current_y = y[batch] # .reshape(batchsize,1)
                # calculate gradient
                gradient = self.gradient(current_X,current_y,batchsize,n_classes)#.reshape((-1, 1))
                self.betas = (1-learning_rate*self.lmda)*self.betas-learning_rate*gradient
            # if test_data:
            #     print("Epoch {}: Measured {}: {}".format(i+1, self.metric, self.evaluate_accuracy(X_test, y_test)))
            #print(self.betas[0])
            if print_epochs:
                # Testing on a small selection of the training data to not slow down too much
                #print("Epoch {}: {} / {} (on training data)".format(j+1, self.evaluate_accuracy(zip(X[accuracy_batch],y[accuracy_batch])), batchsize))
                print("Epoch {} complete".format(i+1))
        #self.betas = betas # storing the betas/weights
        return self.betas

    def learning_schedule(self, t0, t1, t):
        if self.decay:
            return self.t0/(t+self.t1)
        else:
            return self.learning_rate

    def gradient(self, X, y, n, n_classes):
        # calculate gradient  dC/d\beta
        #pred = self.predict(X)
        # make an array of length n_classes with the predicted digit 
        # giving the index where the array has 1. All other entries are 0

        #diff = np.zeros(len(y))
        #for i in range(len(y)):
            # if pred[i] is correct diff is 1
        #    diff[i] = y[i][pred[i]]

        prob = self.softmax(np.dot(X, self.betas))
        diff = (y-prob)
        #print("prob: ", prob)
        #print("gradient: ",diff)
        return  (-1/n)*np.dot(X.T,diff)


    def predict(self, X):
        '''
        Use the trained model to predict the output/classification
        from the input X.
        '''
        prob = self.softmax(np.dot(X, self.betas))
        return np.argmax(prob, axis=1)
        #return [np.argmax(self.softmax(np.dot(self.betas, x))) for x in X]

    def softmax(self, z):
        '''
        numerically stable softmax function avoiding overflow issues
        [8]
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
    #correctly_predicted = sum(int(y == t) for (y, t) in zip(pred,actual))
    for i in range(n):
        if(pred[i]==actual[i]):
            correctly_predicted += 1


    return correctly_predicted/n