# Third-party libraries
import numpy as np
import math
from Global import BLOCK_SIZE
from helpfunctions import intersect, line_intersection

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.fitness = 0
        self.last_checkpoint = 999999
        
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def ray(self, player, view, obstacles, level):
        new_position = player.position + view*3*BLOCK_SIZE
        intersection = player.position
        x = int(player.position[0] / BLOCK_SIZE)
        y = int(player.position[1] / BLOCK_SIZE)
        for _ in range(4):
            if level[y][x] == 1: 
                #TODO: Move to function
                obstacle = obstacles[y][x]
                points = obstacle.vector_list
                minimum = 2*BLOCK_SIZE
                for i in range(len(points)): 
                    if intersect([player.position, new_position], [points[i], points[(i+1)%len(points)]]):
                        intersection = line_intersection([player.position, new_position], 
                                                         [points[i], points[(i+1)%len(points)]])
                        distance = np.linalg.norm((player.position-intersection))
                        if distance < minimum:
                            minimum = distance
                if minimum < 2*BLOCK_SIZE:
                    return minimum
            #TODO: Fix repeated code
            points = [(x*BLOCK_SIZE, y*BLOCK_SIZE),
                      ((x+1)*BLOCK_SIZE, y*BLOCK_SIZE),
                      ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE),
                      (x*BLOCK_SIZE, (y+1)*BLOCK_SIZE)]
            maximum = 0
            for i in range(len(points)): 
                if intersect([player.position, new_position], [points[i], points[(i+1)%len(points)]]):
                    intersection = line_intersection([player.position, new_position], 
                                                     [points[i], points[(i+1)%len(points)]])
                    distance =  np.linalg.norm((player.position-intersection))
                    if distance > maximum:
                        maximum = distance
                        next_block = i
            if next_block == 0:
                y -= 1
            elif next_block == 1:
                x += 1
            elif next_block == 2:
                y += 1
            else:
                x -= 1
        return 2*BLOCK_SIZE

    def run_ai(self, player, obstacles, level):
        player.speed_up = True
        view = np.array([math.cos(player.direction), math.sin(player.direction)])
        view_left = np.array([math.cos(player.direction + math.pi/6), math.sin(player.direction + math.pi/6)])
        view_right = np.array([math.cos(player.direction - math.pi/6), math.sin(player.direction - math.pi/6)])
        view_sharpleft = np.array([math.cos(player.direction + math.pi/2), math.sin(player.direction + math.pi/2)])
        view_sharpright = np.array([math.cos(player.direction - math.pi/2), math.sin(player.direction - math.pi/2)])
        distances = np.array((self.ray(player, view, obstacles, level),
                              self.ray(player, view_left, obstacles, level),
                              self.ray(player, view_right, obstacles, level),
                              self.ray(player, view_sharpleft, obstacles, level),
                              self.ray(player, view_sharpright, obstacles, level),
                              np.linalg.norm(player.speed))).reshape([6,1])
        result = self.feedforward(distances).argmax()
        if result == 0 : 
            player.turn = "left"
        elif result == 1:
            player.turn = "neutral" 
        else:
            player.turn = "right"
                

    
    
    def procreate(self, mutate_chance, random_chance):
        for i in range(len(self.weights)):
            array_shape = self.weights[i].shape
            for y in range(array_shape[0]):
                for x in range(array_shape[1]):
                    odds = np.random.uniform()
                    if odds < mutate_chance:
                        self.weights[i][y,x] += np.random.normal(0, .3)
                    elif odds < mutate_chance + random_chance:
                        self.weights[i][y,x] = np.random.randn()
        

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
