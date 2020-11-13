import NeuralNet as NN
import numpy as np
import warnings

class NeuralNetClassifier(NN.NeuralNet):
    '''
    An extension of the NeuralNet class for classification.
    '''

    def sgd(self, X, y, n_epochs, batchsize, learning_rate, n_classes=10, test_data=None, print_epochs=True):
        '''
        n_classes       - the number of classes/digits to in the dataset
        '''
        if(y.ndim==1): # transform each yi to a vector with length n_classes
           y = np.array([self._vector_transform(yi,n_classes) for yi in y])

        return super().sgd(X=X, y=y, n_epochs=n_epochs, batchsize=batchsize, learning_rate=learning_rate, test_data=test_data, print_epochs=print_epochs)

    def predict(self,X):
        return [np.argmax(self.feedforward(x)) for x in X]

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