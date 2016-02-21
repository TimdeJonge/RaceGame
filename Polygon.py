import pygame
from Vector import Vector
from Rectangle import Rectangle
from Global import *


class Polygon:
    """Used for Obstacles that are polygons."""
    def __init__(self, pointlist, colour=OBSTACLE_COLOUR):
        self.pointlist = pointlist
        self.vector_list = []
        for point in pointlist:
            self.vector_list.append(Vector(point))
        self.colour = colour
        x_min = pointlist[0][0]
        x_max = pointlist[0][0]
        y_min = pointlist[0][1]
        y_max = pointlist[0][1]
        for point in pointlist:
            if point[0] < x_min:
                x_min = point[0]
            if point[0] > x_max:
                x_max = point[0]
            if point[1] < y_min:
                y_min = point[1]
            if point[1] > y_max:
                y_max = point[1]
        self.bounding_box = Rectangle([x_min, y_min], [x_max, y_max])

    def draw(self, screen, camera):
        draw_list = []
        for vector in self.vector_list:
            draw_vector = vector - camera
            draw_list.append(draw_vector.values)
        pygame.draw.polygon(screen, self.colour, draw_list)

    def contains(self, point):
        if not self.bounding_box.contains(point):
            return False
        length = len(self.vector_list)
        # magic for loop: A point is inside the polygon when it's on the same side of an edge as an other vertex.
        # the following for loop checks this for every edge. This works, don't change unless
        # the implementation of a point changes.
        # Dependent on this function: Unit.check_collision
        for i in range(length):
            a = self.vector_list[i]
            b = self.vector_list[(i+1) % length]
            c = self.vector_list[(i+2) % length]
            vector1 = Vector([a.values[1] - b.values[1], b.values[0] - a.values[0]])
            vector2 = point - a
            vector3 = c - a
            if vector1.inner(vector2) * vector1.inner(vector3) < 0:
                return False
        return True

    def handle_collision(self, player):
        new_position = player.position + player.speed
        length = len(self.vector_list)
        point1, point2 = 0, 0
        for i in range(length):
            a = self.vector_list[i]
            b = self.vector_list[(i+1) % length]
            vector1 = Vector([a.values[1] - b.values[1], b.values[0] - a.values[0]])
            vector2 = player.position - a
            vector3 = new_position - a
            if vector1.inner(vector2) * vector1.inner(vector3) < 0:
                point1 = a
                point2 = b
        if point1 == 0 and point2 == 0:
            print("TIME TO TURN ON THE DEBUGGER")
            player.speed = player.speed.scalar(-1)
        parallel, perpendicular = player.speed.split(point1 - point2)
        new_speed = (parallel - perpendicular).scalar(.5)
        player.set_speed(new_speed.values[0], new_speed.values[1])

poly = Polygon([(400, 150), (800, 150), (1000, 300), (1000, 500),
                (800, 650), (400, 650), (200, 500), (200, 300)])
poly.contains(Vector([0, 0]))
