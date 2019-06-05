#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def mutate_weight(self, size):
        array_shape = self.weights[0].shape
        x_change = random.choice(range(0,array_shape[0]))
        y_change = random.choice(range(0,array_shape[1]))
        self.weights[0][x_change][y_change] += random.choice([-1,1])*size
        
        
    def random_weight(self):
        array_shape = self.weights[0].shape
        x_change = random.choice(range(0,array_shape[0]))
        y_change = random.choice(range(0,array_shape[1]))
        self.weights[0][x_change][y_change] = random.uniform(-1,1)
        

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
