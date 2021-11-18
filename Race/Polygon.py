import pygame
from Race.Global import OBSTACLE_COLOUR
import numpy as np


class Polygon:
    """Used for Obstacles that are polygons."""
    def __init__(self, pointlist, colour=OBSTACLE_COLOUR):
        self.pointlist = pointlist
        self.vector_list = []
        for point in pointlist:
            self.vector_list.append(np.array(point))
        self.colour = colour

    def draw(self, screen, camera):
        draw_list = []
        for vector in self.vector_list:
            draw_vector = vector - camera
            draw_list.append(draw_vector)
        pygame.draw.polygon(screen, self.colour, draw_list)
