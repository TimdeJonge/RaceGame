import pygame
from Vector import Vector
from Global import *
import math

class Circle:
    """Used for Obstacles that are polygons."""
    def __init__(self, centre, radius, colour=OBSTACLE_COLOUR):
        self.centre = Vector(centre)
        self.radius = radius
        self.colour = colour

    def draw(self, screen, camera):
        draw_centre = self.centre - camera
        pygame.draw.circle(screen, self.colour, draw_centre.to_int(), self.radius)

    def collides_with(self, point):
        if (self.centre - point).norm() > self.radius:
            return False
        else:
            return True

    def handle_collision(self, player):
        collision_line = Vector([player.position.values[1] - self.centre.values[1],
                                 self.centre.values[0] - player.position.values[0]])
        parallel, perpendicular = player.speed.split(collision_line)
        new_speed = (parallel - perpendicular).scalar(.5)
        player.set_speed(new_speed.values[0], new_speed.values[1])
