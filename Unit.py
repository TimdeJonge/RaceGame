import pygame
import math
from Global import *
from Vector import Vector


class Unit:
    """Used for all moving objects. Defined by position and speed, maybe to change soon. No external dependencies."""

    def __init__(self, position=Vector([0, 0]), speed=Vector([0, 0]), radius=5, colour=PLAYER_COLOUR):
        self.position = position
        self.speed = speed
        self.colour = colour
        self.radius = radius
        self.old_position = position
        if speed.values[0] == 0 and speed.values[1] >= 0:
            self.direction = math.pi / 2
        if speed.values[0] == 0 and speed.values[1] < 0:
            self.direction = - math.pi / 2
        else:
            self.direction = math.atan(speed.values[1] / speed.values[0])

    def move(self, obstacles):
        # TODO: Add particles behind the boat
        for obstacle in obstacles:
            self.check_collision(obstacle)
        self.old_position = self.position
        self.position += self.speed
        self.speed -= self.speed.scalar(.01)

    def check_collision(self, obstacle):
        new_position = self.position + self.speed
        if obstacle.collides_with(new_position):
            # TODO: Add sound effect to collision
            print("Collision!")
            obstacle.handle_collision(self)

    def set_speed(self, x_speed, y_speed):
        self.speed.values[0] = x_speed
        self.speed.values[1] = y_speed

    def accelerate(self):
        acceleration_vector = Vector((math.cos(self.direction), math.sin(self.direction))).scalar(.05)
        self.speed += acceleration_vector

    def brake(self):
        acceleration_vector = Vector((math.cos(self.direction), math.sin(self.direction))).scalar(.05)
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

    def turn(self, angle):
        self.direction += angle

    def draw(self, screen, camera):
        pygame.draw.circle(screen, self.colour, (self.position - camera).to_int(), self.radius)


class Player(Unit):
    def restart(self, position, speed):
        self.position = position
        self.speed = speed
        self.direction = math.atan(speed.values[1] / speed.values[0])

    # TODO: make a sprite for the player with this bounding box, and use that rather than the circle
    def draw_player(self, screen, camera):
        pygame.draw.circle(screen, self.colour, (self.position - camera).to_int(), self.radius)
