import NeuralNet as NN

class NeuralNetClassification(NN.NeuralNet):
    '''
    An extension of the NeuralNet class for classification.
    '''

    def sgd(self, X, y, n_epochs, batchsize, learning_rate, n_classes=10, test_data=None, print_epochs=True):
        '''
        n_classes       - the number of classes/digits to in the dataset
        '''
        if(y.ndim==1): # transform each yi to a vector with length n_classes
           y = np.array([self._vector_transform(yi,n_classes) for yi in y])

        super().sgd(self, X, y, n_epochs, batchsize, learning_rate, test_data=None, print_epochs=True)

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