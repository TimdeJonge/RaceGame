import pygame
import math
import numpy as np
from Global import ACCELERATION_DEFAULT, BLACK, BLOCK_SIZE, RED
from helpfunctions import intersect, split, line_intersection

class Player:
    def __init__(self, position = None, speed= None,
                 radius=5, colour=BLACK, network = None):
        if position is not None:
            self.position = position
        else:
            self.position = np.array([4.5*BLOCK_SIZE, 1.6*BLOCK_SIZE])
        if speed is not None: 
            self.speed = speed
        else: 
            self.speed = np.array([1.0, 0.05])
        self.colour = colour
        self.network = network
        self.radius = radius
        self.turn = "neutral"
        self.speed_up = False
        self.speed_down = False
        self.speed_boost = False
        self.speed_boost_counter = -5000
        self.acceleration = ACCELERATION_DEFAULT
        self.fitness = 0
        self.checkpoint = 1
        self.human = False
        self.last_checkpoint = 999999 
        if self.speed[0] == 0 and self.speed[1] >= 0:
            self.direction = math.pi / 2
        elif self.speed[0] == 0 and self.speed[1] < 0:
            self.direction = - math.pi / 2  
        else:
            self.direction = math.atan(self.speed[1] / self.speed[0])

    def update(self, obstacles, level, checkpoints, counter):     
        if self.turn == "right":
            self.rotate(math.pi/45)
        elif self.turn == "left":
            self.rotate(-math.pi/45)
        if self.speed_up:
            self.accelerate(ACCELERATION_DEFAULT)
        if self.speed_down:
            self.accelerate(-ACCELERATION_DEFAULT)
        self.move(obstacles, level, checkpoints, counter)
        self.fitness += np.linalg.norm(self.speed)

    def observe(self, obstacles, level):
        view = np.array([math.cos(self.direction), math.sin(self.direction)])
        view_left = np.array([math.cos(self.direction + math.pi/6), math.sin(self.direction + math.pi/6)])
        view_right = np.array([math.cos(self.direction - math.pi/6), math.sin(self.direction - math.pi/6)])
        view_sharpleft = np.array([math.cos(self.direction + math.pi/2), math.sin(self.direction + math.pi/2)])
        view_sharpright = np.array([math.cos(self.direction - math.pi/2), math.sin(self.direction - math.pi/2)])
        distances = np.array((self.ray(view, obstacles, level),
                              self.ray(view_left, obstacles, level),
                              self.ray(view_right, obstacles, level),
                              self.ray(view_sharpleft, obstacles, level),
                              self.ray(view_sharpright, obstacles, level),
                              np.linalg.norm(self.speed)))
        return distances

    def ray(self, view, obstacles, level):
        new_position = self.position + view*3*BLOCK_SIZE
        intersection = self.position
        x = int(self.position[0] / BLOCK_SIZE)
        y = int(self.position[1] / BLOCK_SIZE)
        for _ in range(4):
            if level[y][x] == 1: 
                #TODO: Move to function
                obstacle = obstacles[y][x]
                points = obstacle.vector_list
                minimum = 2*BLOCK_SIZE
                for i in range(len(points)): 
                    if intersect([self.position, new_position], [points[i], points[(i+1)%len(points)]]):
                        intersection = line_intersection([self.position, new_position], 
                                                         [points[i], points[(i+1)%len(points)]])
                        distance = np.linalg.norm((self.position-intersection))
                        if distance < minimum:
                            minimum = distance
                if minimum < 2*BLOCK_SIZE:
                    return 1 - (minimum / (2*BLOCK_SIZE)) 
            #TODO: Fix repeated code
            points = [(x*BLOCK_SIZE, y*BLOCK_SIZE),
                      ((x+1)*BLOCK_SIZE, y*BLOCK_SIZE),
                      ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE),
                      (x*BLOCK_SIZE, (y+1)*BLOCK_SIZE)]
            maximum = 0
            for i in range(len(points)): 
                if intersect([self.position, new_position], [points[i], points[(i+1)%len(points)]]):
                    intersection = line_intersection([self.position, new_position], 
                                                     [points[i], points[(i+1)%len(points)]])
                    distance =  np.linalg.norm((self.position-intersection))
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
        return 0

    def move(self, obstacles, level, checkpoints, counter):
        new_position = self.position + self.speed + self.speed/np.linalg.norm(self.speed)*5
        x = int(new_position[0]//BLOCK_SIZE)
        y = int(new_position[1]//BLOCK_SIZE)
        if level[y][x]: 
            self.player_collision(obstacles[y][x], new_position)
        self.check_point(checkpoints, counter)
        self.position = self.position + self.speed
        self.friction()
        #self.check_food(food)
            
    def check_point(self, checkpoints, counter):
        new_position = self.position + self.speed
        if intersect([self.position, new_position], checkpoints[self.checkpoint]):
            self.fitness += 10000
            self.last_checkpoint = counter
            self.checkpoint = (self.checkpoint + 1) % len(checkpoints)

    def accelerate(self, pace):
        acceleration = np.array([math.cos(self.direction), math.sin(self.direction)])*pace
        self.speed += acceleration

    def rotate(self, angle):
        self.direction += angle

    def player_collision(self, obstacle, new_position):
        '''Checks whether the player collides with the given obstacle.
        If so, sets the player's parameters correctly for completing the bounce
        Requires obstacle player can bounce again, position after 1 timestep. '''
        points = obstacle.vector_list
        for i in range(len(points)): 
            if intersect([self.position, new_position], [points[i], points[(i+1)%len(points)]]):
                parallel, perpendicular = split(points[i] - points[(i+1)%len(points)], self.speed)
                new_speed = (parallel - perpendicular)*.3
                self.set_speed(new_speed[0], new_speed[1])
                if np.linalg.norm(self.speed) > 1:
                    self.fitness -= 100 

    def friction(self):
        if np.linalg.norm(self.speed) < 5:
            self.speed -= self.speed*0.05
        else:
            self.speed -= self.speed*.0125*np.linalg.norm(self.speed)       
        
    def set_speed(self, x_speed, y_speed):
        self.speed[0] = x_speed
        self.speed[1] = y_speed


    def restart(self, position, speed):
        self.position = position
        self.speed = speed
        self.direction = math.atan(speed[1] / speed[0])

    def draw_player(self, screen, camera):
        pygame.draw.circle(screen, self.colour, (self.position - camera).astype(int), self.radius)
        standard_vector = np.array([math.cos(self.direction), math.sin(self.direction)])
        pygame.draw.line(screen, self.colour, (self.position - camera).astype(int),
                                              (self.position + standard_vector*(10) - camera).astype(int))
