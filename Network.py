#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, a):
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
