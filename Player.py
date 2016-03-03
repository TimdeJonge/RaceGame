import pygame
import math
from Global import *
from Vector import Vector


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

    def run_ai(self, level):
        self.speed_up = True
        new_position = self.position + self.speed.scalar(100)
        good_choice = True
        for obstacle in level:
            if obstacle.contains(new_position):
                good_choice = False
                break
        if not good_choice:
            self.turn = "right"
        else:
            self.turn = "neutral"

    def update(self, counter, level):
        if not self.human:
            self.run_ai(level)

        if self.human == "test":
            if self.speed.values[0] <= 0:
                print(self.position, self.speed, self.direction)
                self.set_speed(0, 1)
                self.speed_up = False
                self.turn = "neutral"
            else:
                self.speed_up = True
                if self.direction < math.pi:
                    self.turn = "right"
                else:
                    self.turn = "neutral"

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
            self.speed_boost_counter = counter
            self.acceleration = 3*ACCELERATION_DEFAULT
            self.speed_boost = False
        if counter - self.speed_boost_counter == .5*FRAME_RATE:
            self.acceleration = ACCELERATION_DEFAULT

    def move(self, obstacles):
        # TODO: Add particles behind the boat
        for obstacle in obstacles:
            self.check_collision(obstacle)
        self.position += self.speed
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

    def accelerate(self):
        acceleration_vector = Vector((math.cos(self.direction), math.sin(self.direction))).scalar(self.acceleration)
        self.speed += acceleration_vector

    def brake(self):
        acceleration_vector = Vector((math.cos(self.direction), math.sin(self.direction))).scalar(self.acceleration)
        self.speed -= acceleration_vector

    def old_turn(self, angle):
        # In other words: Make the controls less wonky
        # It is possible to move this to the Vector class, but since the Vector class might be used for Vectors
        # with dimension > 3, this is undesirable.
        if self.speed.values != [0, 0]:
            cos_theta = math.cos(angle)
            sin_theta = math.sin(angle)
            x_speed = (cos_theta*self.speed.values[0] - sin_theta*self.speed.values[1])
            y_speed = (sin_theta*self.speed.values[0] + cos_theta*self.speed.values[1])
            self.speed.values[0] = x_speed
            self.speed.values[1] = y_speed

    def rotate(self, angle):
        self.direction += angle

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
