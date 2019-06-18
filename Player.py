import pygame
import math
import numpy as np
from Network import Network
from Global import ACCELERATION_DEFAULT, PLAYER_COLOUR, FRAME_RATE, BLOCK_SIZE
from Vector import Vector
from Rectangle import Rectangle

class Player:
    def __init__(self, sounds, position=Vector([0, 0]), speed=Vector([0, 0]), radius=5, colour=PLAYER_COLOUR, human=False):
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
        if speed.values[0] == 0 and speed.values[1] >= 0:
            self.direction = math.pi / 2
        if speed.values[0] == 0 and speed.values[1] < 0:
            self.direction = - math.pi / 2
        else:
            self.direction = math.atan(speed.values[1] / speed.values[0])
        self.bump = sounds
        if not self.human:
            self.network = Network([6,4,3])

    def run_ai(self, counter, level):
        self.speed_up = True
        view = Vector([math.cos(self.direction), math.sin(self.direction)])
        view_right = Vector([math.cos(self.direction + math.pi/6), 
                            math.sin(self.direction + math.pi/6)])
        view_left = Vector([math.cos(self.direction - math.pi/6), 
                            math.sin(self.direction - math.pi/6)])
        view_hardright = Vector([math.cos(self.direction + 2*math.pi/3), 
                            math.sin(self.direction + 2*math.pi/3)])
        view_hardleft = Vector([math.cos(self.direction - 2*math.pi/3), 
                            math.sin(self.direction - 2*math.pi/3)])
    
        views = [view_hardleft, view_left, view, view_right, view_hardright]
        if counter%5 == 0:
            distances = [200 for _ in range(5)]
            i = 5
            while (i < 100): 
                for j in range(len(views)):
                    if i < distances[j]:
                        new_position = self.position + views[j].scalar(i)
                        for obstacle in level:
                            if obstacle.contains(new_position):
                                distances[j] = i 
                i+= 5
            distances.append(self.speed.norm())
            result = self.network.feedforward(np.reshape(distances, (6,1))).argmax()
            print(result)
            if result == 0 : 
                self.turn = "left"
            elif result == 1:
                self.turn = "neutral" 
            else:
                self.turn = "right"

    def update(self, counter, level):
        if not self.human:
            self.run_ai(counter, level)
              
        #Handling of human input            
        if self.turn == "right":
            self.rotate(math.pi/120)
        elif self.turn == "left":
            self.rotate(-math.pi/120)
        if self.speed_up:
            if self.speed.norm() != 0:
                self.accelerate()
        if self.speed_down:
            if self.speed.norm() != 0:
                self.brake()
        if self.speed_boost:
            self.speed_boost_ounter = counter
            self.acceleration = 3*ACCELERATION_DEFAULT
            self.speed_boost = False
        if counter - self.speed_boost_counter == .5*FRAME_RATE:
            self.acceleration = ACCELERATION_DEFAULT

    def accelerate(self):
        acceleration_vector = Vector((math.cos(self.direction), math.sin(self.direction))).scalar(self.acceleration)
        self.speed += acceleration_vector

    def brake(self):
        acceleration_vector = Vector((math.cos(self.direction), math.sin(self.direction))).scalar(self.acceleration)
        self.speed -= acceleration_vector

    def rotate(self, angle):
        self.direction += angle
        
    def move(self, obstacles, level):
        # TODO: Add particles behind the boat
        new_position = self.position + self.speed
        x = int(new_position.values[0]//BLOCK_SIZE)
        y = int(new_position.values[1]//BLOCK_SIZE)
        if level[x][y] == 1: 
            #TODO: Base on line rather than object
            Rectangle((x*BLOCK_SIZE, y*BLOCK_SIZE), ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE)).handle_collision(self)
        self.position = self.position + self.speed
        if self.speed.norm() < 5:
            self.speed -= self.speed.scalar(0.01)
        else:
            self.speed -= self.speed.scalar(.0025*self.speed.norm())
        

    def check_collision(self, obstacle):
        new_position = self.position + self.speed
        if obstacle.contains(new_position):
            volume = min(1,obstacle.handle_collision(self)/5)
            self.bump.set_volume(volume)
            self.bump.play()

    def set_speed(self, x_speed, y_speed):
        self.speed.values[0] = x_speed
        self.speed.values[1] = y_speed


    def restart(self, position, speed):
        self.position = position
        self.speed = speed
        self.direction = math.atan(speed.values[1] / speed.values[0])

    # TODO: make a sprite for the player with this bounding box, and use that rather than the circle
    def draw_player(self, screen, camera):
        pygame.draw.circle(screen, self.colour, (self.position - camera).to_int(), self.radius)
        standard_vector = Vector((math.cos(self.direction), math.sin(self.direction)))
        pygame.draw.line(screen, self.colour, (self.position - camera).values,
                                              (self.position + standard_vector.scalar(10) - camera).values)
