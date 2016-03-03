from Vector import Vector
from Global import *
import pygame


class Rectangle:
    def __init__(self, point1, point2, colour=OBSTACLE_COLOUR):
        self.point1 = Vector(point1)
        self.point2 = Vector(point2)
        self.colour = colour
        self.left = min(self.point1.values[0], self.point2.values[0])
        self.up = min(self.point1.values[1], self.point2.values[1])
        self.right = max(self.point1.values[0], self.point2.values[0])
        self.down = max(self.point1.values[1], self.point2.values[1])

    def contains(self, point):
        return self.left <= point.values[0] <= self.right and self.up <= point.values[1] <= self.down

    def handle_collision(self, player):
        new_position = player.position + player.speed
        if (player.position.values[0] <= self.left <= new_position.values[0] or
                new_position.values[0] <= self.right <= player.position.values[0]):
            direction = "y"
        else:
            direction = "x"
        if direction == "x":
            parallel = Vector([player.speed.values[0], 0])
            perpendicular = Vector([0, player.speed.values[1]])
        else:
            perpendicular = Vector([player.speed.values[0], 0])
            parallel = Vector([0, player.speed.values[1]])
        new_speed = (parallel - perpendicular).scalar(.5)
        volume = (player.speed - new_speed).norm()
        player.set_speed(new_speed.values[0], new_speed.values[1])
        return volume

    def draw(self, screen, camera):
        draw_rect = [0, 0]
        draw_rect[0] = (Vector([self.left, self.up]) - camera).values
        draw_rect[1] = [self.right - self.left, self.down - self.up]
        pygame.draw.rect(screen, self.colour, draw_rect)
