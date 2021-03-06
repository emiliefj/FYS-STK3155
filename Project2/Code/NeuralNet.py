import numpy as np
import warnings

class NeuralNet():
    """
    A feed forward neural network for regression with a flexible 
    number of nodes in each layer, and a flexible number of 
    hidden layers.
    Stochastic gradient descent with backpropagation is used to 
    train the model.
    Readability is prioritized over speed.
    
    inspired by http://neuralnetworksanddeeplearning.com/

    layers_list - the number of nodes per layer in an ordered list
    h_af        - the activation function used for the hidden neurons
                  available options: 
                  - 'sigmoid': logistic sigmoid function
                  - 'relu': ReLU - Rectified Linear Unit
                  - 'leaky': leaky ReLU
                  - 'elu'
    o_af        - the activation function for the output neurons
                  available options
                  - 'none': No activation function in output layer.
                            Typically used for regression
                  - 'sigmoid': logistic sigmoid function
                  - 'softmax':
                  - 'relu'
                  - 'leaky'
                  - 'step': binary step function
    cost        - the cost function used in learning: 
                  - 'cross-entropy', or 
                  - 'squared-loss'
    lmda        - regularization parameter for l2 regularization
    seed        - seed seed for random function. Used for reproduceability.
    """
    def __init__(self, layers_list=[64,30,10], h_af='sigmoid', o_af='softmax', cost='cross-entropy', lmda=0, metric='accuracy',seed=839):
        np.random.seed(seed)
        self.n_layers = len(layers_list)  # number of layers
        self.lmda = lmda
        self.biases = [np.random.randn(y, 1) for y in layers_list[1:]] # no bias to first/input layer
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(layers_list[:-1], layers_list[1:])]
        self.metric = metric.lower()

        # Choice of activation function
        self.h_af = h_af.lower()
        self.o_af = o_af.lower()
        # Choice of cost function
        self.cost = cost.lower()

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        if(a.ndim<2):
            a = a.reshape((-1, 1))
        # loop through hidden layers
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = self.activation_function(np.dot(w, a)+b)
        # output layer
        z = np.dot(self.weights[-1], a)+self.biases[-1]
        return self.o_activation_function(z)
        
    def sgd(self, X, y, n_epochs, batchsize, learning_rate, test_data=None, print_epochs=True): # Add possibility for decreasing step size?
        """
        Performs stochastic gradient descent to train the model. Loops over 
        n_epochs and in each loop performs one step for each (randomly selected,
        with replacement) batch of the input X,y. Uses backpropagation to 
        update weights and biases. The learning rate sets the step size.
        Prints a line at the end of each epoch to keep track of progress.

        X               - array of inputs
        y               - array of targets
        n_epochs        - the number of epochs to train the model for
        learning_rate   - the step size gamma
        test_data       - allows the ability to add a separate set of
                          test data to monitor progress of the model
                          for each epoch
        print_epochs    - if True a line is printed at the end of every 
                          epoch to track progress
        """
        if(test_data):
            X_test, y_test = zip(*test_data)
            n_test = len(X_test)
        #if(y.ndim==1): # only for classification of 10 things. Make optional, and more general (i.e. varying number of outputs)
        #    y = np.array([self._vector_transform(yi) for yi in y])
        n = len(X)
        n_batches = int(n/batchsize)
        accuracy_batch = np.random.randint(n,size=batchsize)
        for j in range(n_epochs):
            for b in range(n_batches):
                # fetch X and y of current mini-batch
                batch = np.random.randint(n,size=batchsize)
                current_X = X[batch]
                current_y = y[batch]#.reshape(batchsize,1)
                # print(np.shape(current_X))
                # print(np.shape(current_y))
                self.update_batch(current_X, current_y, learning_rate)
            if test_data:
                print("Epoch {}: Measured {}: {}".format(j+1, self.metric, self.evaluate_accuracy(X_test, y_test)))
            elif print_epochs:
                # Testing on a small selection of the training data to not slow down too much
                #print("Epoch {}: {} / {} (on training data)".format(j+1, self.evaluate_accuracy(zip(X[accuracy_batch],y[accuracy_batch])), batchsize))
                print("Epoch {} complete".format(j+1))
                

    def update_batch(self, X_batch, y_batch, lr):
        """
        Update the network's weights and biases by applying gradient
        descent using backpropagation on the single mini batch given
        as input.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        n = len(X_batch)
        for i in range(n):
            db_list, dw_list = self.backprop(X_batch[i], y_batch[i])
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, db_list)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, dw_list)]
        self.weights = [(1-lr*(self.lmda/n))*w-(lr/n)*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(lr/n)*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """ Perform backpropagation to find change in weights and biases

        Performs feedforward computing output at each layer. Then
        computes the error in the output layer and propagates
        back through the layer calculating output error. Finds
        the change dw and db for each weight and bias, aka dC/dw
        and dC/db. 

        x, y    - input and output 
        """
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))

        # feedforward #
        activation = x    # activation of input layer
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        s = 0
        for b, w in zip(self.biases[:-1], self.weights[:-1]): # all but last layer (output layer)
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
        # activation in output layer
        z = np.dot(self.weights[-1], activation)+self.biases[-1]
        zs.append(z)
        activation = self.o_activation_function(z)
        activations.append(activation)
        # backward pass #
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        delta = self.calculate_delta(zs[-1], activation, y)
        # output error delta^L in output layer
        db[-1] = delta
        dw[-1] = np.dot(delta, activations[-2].T)
        # output error delta^l in remaining layers
        for l in range(2, self.n_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].T, delta)*self.h_af_derivative(z)
            db[-l] = delta
            dw[-l] = np.dot(delta, activations[-l-1].T)
        return (db, dw)

    def predict(self,X):
        return [self.feedforward(x) for x in X]
 

    def evaluate_accuracy(self, X, y):
        pred = [self.feedforward(x) for x in X]
        if self.metric=='accuracy':
            pred = self.predict(X)
            return accuracy(pred,y)
        if self.metric=='mse':
            return mse(pred,y)
        if self.metric=='r2':
            return r2(pred,y)
        else:
            warnings.warn('Could not recognize chosen metric. No metric used.')
            return 0


    def calculate_delta(self, z, a, y):
        y = y.reshape((-1, 1))
        if self.cost=="cross-entropy":
            return (a-y)
        if self.cost=="squared_error":
            return (a-y)*self.o_af_derivative(z)
        else: 
            warnings.warn('Could not recognize chosen cost function. \
                Using default \'cross-entropy\'.')
            return (a-y)


    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives 
        (\\partial C_x/\\partial) a for the output activations.

        Todo: open for different cost function
        """
        return (output_activations-y)


    # Activation functions #

    def activation_function(self, z):
        ''' 
        Calculating the output of the node in a hidden layer using the
        chosen activation function 
        
        z   - The input to the activation function wa+b
        '''
        if self.h_af=='sigmoid':
            return self.sigmoid(z)
        if self.h_af=='elu':
            return self.elu(z)
        if self.h_af=='relu':
            return self.relu(z)
        if self.h_af=='leaky':
            return self.leaky_relu(z)
        else: 
            warnings.warn('Did not recognize chosen activation function \
                for the hidden layers. Using default sigmoid function')
            return self.sigmoid(z)

    def o_activation_function(self, z):
        ''' 
        Calculating the output of the output layer using the chosen 
        activation function for the output layer.

        z   - input to the activation function, wa+b
        '''
        if self.o_af=='none':
            return z
        if self.o_af=='sigmoid':
            return self.sigmoid(z)
        if self.o_af=='softmax':
            return self.softmax(z)
        if self.o_af=='relu':
            return self.relu(z)
        if self.o_af=='leaky':
            return self.leaky_relu(z)
        if self.o_af=='step':
            return self.step(z)
        else: 
            warnings.warn('Did not recognize chosen activation function \
                for the output layer. Using default sigmoid function')
            return self.sigmoid(z)


    def tanh(self, z):
        return np.tanh(z)

    def relu(self, z):
        #return np.array([max(0,zi) for zi in z])
        return np.where(z>0, z, 0)

    def leaky_relu(self, z):
        return np.where(z>0, z, 0.01*z)

    def elu(self, z):
        ''' ELU with alpha=1 '''
        return  np.where(z<0, np.exp(z)-1, z)

    def sigmoid(self, z):
        ''' logistic sigmoid function '''
        #return np.where(z<0, np.exp(z)/(1+np.exp(z)), np.exp(-z)/(1+np.exp(-z)))
        return 1./(1+np.exp(-z))

    def step(self,z):
        return np.where(z>0, 1, 0)

    def softmax(self, z):
        return np.exp(z)/np.sum(np.exp(z)) 


    def o_af_derivative(self,z):
        '''
        The derivatrive of the chosen activation function for the output 
        layer

        z   - input to the derivative of the activation function, wa+b
        '''
        if self.o_af=='none':
            return 1.0
        if self.o_af=='sigmoid':
            return self.d_sigmoid(z)
        if self.o_af=='relu':
            return self.d_relu(z)
        if self.o_af=='leaky':
            return self.d_leaky_relu(z)
        if self.o_af=='elu':
            return self.d_elu(z)
        if self.o_af=='step':
            return self.d_step(z)
        else: # default
            return self.d_sigmoid(z)

    def h_af_derivative(self,z):
        '''
        The derivatrive of the chosen activation function for the hidden 
        layers

        z   - input to the derivative of the af, wa+b
        '''
        if self.h_af=='sigmoid':
            return self.d_sigmoid(z)
        if self.h_af=='relu':
            return self.d_relu(z)
        if self.h_af=='leaky':
            return self.d_leaky_relu(z)
        else: # default
            return self.d_sigmoid(z)


    def d_tanh(self, z):
        return 1-np.tanh(z)**2
        
    def d_relu(self, z):
        return np.where(z>0, 1, 0)

    def d_leaky_relu(self, z):
        return np.where(z<0, 1, 0.01)

    def d_elu(self, z):
        return np.where(z<0, np.exp(z), z)

    def d_sigmoid(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def d_step(self,z):
        '''
        The step function has a discontinuos derivative at 0, and is 
        zero elsewhere. Setting it to zero everywhere to avoid 
        discontinuity issues.
        '''
        return 0

    def _vector_transform(self,i):
        ''''
        Transform input i to a 10d vector with 1. on the ith index (the
        index corresponding to the number-value.
        '''
        vector = np.zeros((10,))
        vector[i] = 1.0
        return vector




def mse(pred, actual):
    '''
    The mean squared error. Used to measure performance on 
    regression problems

    mse = sum((actual-pred)^2)/n

    pred    - the prediction made by the model
    actual  - the actual value in the data aka target
    '''
    pred = np.array(pred)
    actual = np.array(actual)
    if(pred.shape!=actual.shape):
        pred = pred.reshape((-1, 1))
        actual = actual.reshape((-1, 1))

    return np.mean((actual-pred)**2)

def r2(pred, actual):
    '''
    The r2 score. Used to measure performance on 
    regression problems

    r2 = RSS/TSS

    pred    - the prediction made by the model
    actual  - the actual value in the data aka target
    '''
    pred = np.array(pred) # skip this
    actual = np.array(actual)
    if(pred.shape!=actual.shape):
        pred = pred.reshape((-1, 1))
        actual = actual.reshape((-1, 1))
    return 1-np.sum((actual-pred)**2)/np.sum((actual-np.mean(actual))**2)

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
