import pygame
import math
import numpy as np
from Network import Network
from Global import ACCELERATION_DEFAULT, PLAYER_COLOUR, FRAME_RATE, BLOCK_SIZE
from Rectangle import Rectangle
from helpfunctions import intersect, line_intersection, split 

class Player:
    def __init__(self, sounds, position=np.array([0, 0]), speed=np.array([0, 0]), radius=5, colour=PLAYER_COLOUR, human=False):
        self.position = position
        self.speed = speed
        self.colour = colour
        self.human = human
        self.radius = radius
        self.turn = "neutral"
        self.speed_up = False
        self.speed_down = False
        self.speed_boost = False
        self.speed_boost_counter = -5000
        self.acceleration = ACCELERATION_DEFAULT
        if speed[0] == 0 and speed[1] >= 0:
            self.direction = math.pi / 2
        elif speed[0] == 0 and speed[1] < 0:
            self.direction = - math.pi / 2
        else:
            self.direction = math.atan(speed[1] / speed[0])
        self.bump = sounds
        if not self.human:
            self.network = Network([2,4,3])

    def run_ai(self, obstacles, level):
        self.speed_up = True
        view = np.array([math.cos(self.direction), math.sin(self.direction)])
        distances = (self.ray(view, obstacles, level), np.linalg.norm(self.speed))
        result = self.network.feedforward(np.reshape(distances, (2,1))).argmax()
        if result == 0 : 
            self.turn = "left"
        elif result == 1:
            self.turn = "neutral" 
        else:
            self.turn = "right"
                
    def ray(self, view, obstacles, level):
        new_position = self.position + view*2*BLOCK_SIZE
        intersection = self.position
        x = int(self.position[0] / BLOCK_SIZE)
        y = int(self.position[1] / BLOCK_SIZE)
        for _ in range(4):
            if level[x][y] == 1: 
                obstacle = obstacles[x][y]
                points = obstacle.vector_list #CHECK
                minimum = 2*BLOCK_SIZE
                for i in range(len(points)): 
                    if intersect([self.position, new_position], [points[i], points[(i+1)%len(points)]]):
                        intersection = line_intersection([self.position, new_position], 
                                                         [points[i], points[(i+1)%len(points)]])
                        distance = np.linalg.norm((self.position-intersection))
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
        return 2*BLOCK_SIZE
    
    
    
    def update(self, obstacles, level):
        if not self.human:
            self.run_ai(obstacles, level)
              
        #Handling of human input            
        if self.turn == "right":
            self.rotate(math.pi/60)
        elif self.turn == "left":
            self.rotate(-math.pi/60)
        if self.speed_up:
            self.accelerate()
        if self.speed_down:
            self.brake()
# =============================================================================
#         if self.speed_boost:
#             self.speed_boost_counter = counter
#             self.acceleration = 3*ACCELERATION_DEFAULT
#             self.speed_boost = False
#         if counter - self.speed_boost_counter == .5*FRAME_RATE:
#             self.acceleration = ACCELERATION_DEFAULT
# =============================================================================

    def accelerate(self):
        acceleration = np.array([math.cos(self.direction), math.sin(self.direction)])*self.acceleration
        self.speed += acceleration

    def brake(self):
        acceleration = np.array([math.cos(self.direction), math.sin(self.direction)])*self.acceleration
        self.speed -= acceleration

    def rotate(self, angle):
        self.direction += angle
        
    def move(self, obstacles, level):
        # TODO: Add particles behind the boat
        new_position = self.position + self.speed + self.speed/np.linalg.norm(self.speed)*5
        x = int(new_position[0]//BLOCK_SIZE)
        y = int(new_position[1]//BLOCK_SIZE)
        if level[x][y] == 1: 
            obstacle = obstacles[x][y]
            points = obstacle.vector_list #CHECK 
            for i in range(len(points)): 
                if intersect([self.position, new_position], [points[i], points[(i+1)%len(points)]]):
                    #TODO: Check whether this worked
                    parallel, perpendicular = split(points[i] - points[(i+1)%len(points)], self.speed)
                    new_speed = (parallel - perpendicular)*.3
                    self.set_speed(new_speed[0], new_speed[1])
        self.position = self.position + self.speed
        if np.linalg.norm(self.speed) < 5:
            self.speed -= self.speed*0.01
        else:
            self.speed -= self.speed*.0025*np.linalg.norm(self.speed)
        
        """DEAD CODE """
    def check_collision(self, obstacle):
        new_position = self.position + self.speed
        if obstacle.contains(new_position):
            volume = min(1,obstacle.handle_collision(self)/5)
            self.bump.set_volume(volume)
            self.bump.play()

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
