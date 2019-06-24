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
    
    def procreate(self, mutate_chance, random_chance):
        for i in range(len(self.weights)):
            array_shape = self.weights[i].shape
            for y in range(array_shape[0]):
                for x in range(array_shape[1]):
                    odds = np.random.uniform()
                    if odds < mutate_chance:
                        self.weights[i][y,x] += np.random.uniform() - 0.5
                    elif odds < mutate_chance + random_chance:
                        self.weights[i][y,x] = np.random.randn()
        
        

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
