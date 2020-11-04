import numpy as np

class NeuralNet():
    """docstring for NeuralNet

    inspired by http://neuralnetworksanddeeplearning.com/

    """
    def __init__(self, n_input, n_output, n_hidden=[4,4], h_af='sigmoid', o_af='softmax', seed=839):
        np.random.seed(seed)
        self.n_input = n_input # nodes in input layer
        self.n_layers = len(n_hidden)
        self.biases = [np.random.randn(y, 1) for y in n_hidden]
        self.weights = [np.random.randn(y, x) for x, y in zip(n_output, n_hidden)]

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
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
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

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

    def backprop(self, x, y):   # Rewrite this 
                                # different activation function for output
        """Return a tuple '(nabla_b, nabla_w)' representing the
        gradient for the cost function C_x.  'nabla_b' and
        'nabla_w' are layer-by-layer lists of numpy arrays, similar
        to 'self.biases' and 'self.weights'."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y)*self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        # note on indexing: l = 1 means the last layer of neurons, 
        # l = 2 is the second-last layer, and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives 
        \partial C_x/\partial a for the output activations.
        """
        return (output_activations-y)

    def sgd_mine(self):
        n_batches = int(len(self.X)/self.batchsize)
        beta_0 = np.random.randn(len(self.X[0]),1) # Make initial guess for beta
        for i in range(self.n_epochs):
            for b in range(n_batches):
                # fetch X and y of current mini-batch
                batch = np.random.randint(self.n,size=self.batchsize)
                current_X = self.X[batch]
                current_z = self.z[batch].reshape(self.batchsize,1)
                # calculate gradient
                gradient = (current_X.T.dot(current_X.dot(beta_0)-current_z))*2/self.batchsize-self.alpha*(beta_0)
                beta_0 = beta_0-self.learning_schedule(i*n_batches+b)*gradient 
        return beta_0

    def sigmoid(self, z):
        ''' logistic sigmoid function '''
        return 1./(1+exp(-z))

    def d_sigmoid(self, z):
        """Derivative of the sigmoid function."""
        return sigmoid(z)*(1-sigmoid(z))

    def softmax(self, z):
        np.exp(z)/np.sum(np.exp(z)) 