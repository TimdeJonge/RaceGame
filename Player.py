import pygame
import math
import numpy as np
from Global import ACCELERATION_DEFAULT, BLACK, BLOCK_SIZE, RED
from helpfunctions import intersect, split 

class Player:
    def __init__(self, position = None, speed= None,
                 radius=5, colour=BLACK):
        if position is not None:
            self.position = position
        else:
            self.position = np.array([4.5*BLOCK_SIZE, 1.6*BLOCK_SIZE])
        if speed is not None: 
            self.speed = speed
        else: 
            self.speed = np.array([1.0, 0.05])
        self.colour = colour
        self.radius = radius
        self.turn = "neutral"
        self.speed_up = False
        self.speed_down = False
        self.speed_boost = False
        self.speed_boost_counter = -5000
        self.acceleration = ACCELERATION_DEFAULT
        self.fitness = 0
        self.checkpoint = 2
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
            self.rotate(math.pi/60)
        elif self.turn == "left":
            self.rotate(-math.pi/60)
        if self.speed_up:
            self.accelerate(ACCELERATION_DEFAULT)
        if self.speed_down:
            self.accelerate(-ACCELERATION_DEFAULT)
        self.move(obstacles, level, checkpoints, counter)


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
            self.fitness += 1
            self.last_checkpoint = counter
            print(self.fitness)
            self.checkpoint = (self.checkpoint + 1) % len(checkpoints)

    
    
    
    
    def accelerate(self, pace):
        acceleration = np.array([math.cos(self.direction), math.sin(self.direction)])*pace
        self.speed += acceleration


    def rotate(self, angle):
        self.direction += angle


    """Checks whether the player collides with the given obstacle.
    If so, sets the player's parameters correctly for completing the bounce
    Requires obstacle player can bounce again, position after 1 timestep. """ 
    def player_collision(self, obstacle, new_position):
        points = obstacle.vector_list
        for i in range(len(points)): 
            if intersect([self.position, new_position], [points[i], points[(i+1)%len(points)]]):
                parallel, perpendicular = split(points[i] - points[(i+1)%len(points)], self.speed)
                new_speed = (parallel - perpendicular)*.3
                self.set_speed(new_speed[0], new_speed[1])
    
    def friction(self):
        if np.linalg.norm(self.speed) < 5:
            self.speed -= self.speed*0.05
        else:
            self.speed -= self.speed*.0125*np.linalg.norm(self.speed)

    def check_food(self, food):
        if np.linalg.norm(self.position - food.position) < food.radius:
            self.fitness += 1
            food.pickup()
            self.colour = RED
        
        
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
