import numpy as np

class NeuralNet():
    """docstring for NeuralNet



    inspired by http://neuralnetworksanddeeplearning.com/

    """
    def __init__(self, layers_list=[4,4], h_af='sigmoid', o_af='softmax', seed=839):
        np.random.seed(seed)
        self.n_layers = len(layers_list)  # number of layers
        self.biases = [np.random.randn(y, 1) for y in layers_list[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers_list[:-1], layers_list[1:])]

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # Choice of activation function
        self.h_af = h_af.lower()
        self.o_af = o_af.lower()

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function(np.dot(w, a)+b)
        return a
        
    def sgd(self, X, y, n_epochs, batchsize, learning_rate,test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        n_batches = int(len(self.X)/self.batchsize)
        for j in range(n_epochs):
            for b in range(n_batches):
                # fetch X and y of current mini-batch
                batch = np.random.randint(n,size=self.batchsize)
                current_X = X[batch]
                current_z = y[batch].reshape(self.batchsize,1)
                self.update_batch(current_X, current_y, learning_rate)
            print("Epoch {} complete".format(j))

    def update_batch(self, X_batch, y_batch, learning_rate):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for i in len(X_batch):
            delta_nabla_b, delta_nabla_w = self.backprop(X_batch[i], y_batch[i])
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learning_rate/len(X_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(X_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """ Perform backpropagation to find change in weights and biases

        Performs feedforward computing output at each layer. Then
        computes the error in the outputlayer and propagates
        back trhough the layer calculating output error. Finds
        the change dw and db for each weight and bias. 

        x, y    - input and output 
        """
        # feedforward #
        activation = x    # activation of input layer
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]): # all but last layer
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
        # activation in output layer
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = self.o_activation_function(z)
        activations.append(activation)
        # backward pass #
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        delta = self.cost_derivative(activations[-1], y)*self.o_af_derivative(zs[-1])
        # output error in output layer
        db[-1] = delta
        dw[-1] = np.dot(delta, activations[-2].T)
        # output error in remaining layers
        for l in range(2, self.n_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].T, delta)*self.h_af_derivative(z)
            db[-l] = delta
            dw[-l] = np.dot(delta, activations[-l-1].T)
        return (db, dw)


    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives 
        \partial C_x/\partial a for the output activations.

        Todo: open for different cost function
        """
        return (output_activations-y)

    # Activation functions

    def activation_function(self, z):
        ''' calculating the output using the chosen activation function '''
        if self.h_af=='relu':
            return self.relu(z)
        if self.h_af=='leaky':
            return self.leaky_relu(z)
        else: # default
            return self.sigmoid(z)

    def o_activation_function(self, z):
        ''' 
        Calculating the output using the chosen activation 
        function for the output layer
        '''
        if self.o_af=='relu':
            return self.relu(z)
        if self.o_af=='leaky':
            return self.leaky_relu(z)
        if self.o_af=='softmax':
            return self.softmax(z)
        else: # default
            return self.sigmoid(z)

    def relu(self, z):
        return [max(0,zi) for zi in z]

    def leaky_relu(self, z):
        return np.where(z>0, z, 0.01*z)

    def sigmoid(self, z):
        ''' logistic sigmoid function '''
        return 1./(1+exp(-z))

    def softmax(self, z):
        np.exp(z)/np.sum(np.exp(z)) 


    def o_af_derivative(self,z):
        if self.o_af=='relu':
            return self.d_relu(z)
        if self.o_af=='leaky':
            return self.d_leaky_relu(z)
        else: # default
            return self.d_sigmoid(z)

    def h_af_derivative(self,z):
        if self.h_af=='relu':
            return self.d_relu(z)
        if self.h_af=='leaky':
            return self.d_leaky_relu(z)
        else: # default
            return self.d_sigmoid(z)

    def d_sigmoid(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))

    