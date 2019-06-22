import numpy as np

def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])


def intersect(segment1,segment2):
    [A,B] = segment1
    [C,D] = segment2
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def normalize(vector):
    return vector/np.linalg.norm(vector)

def projects(vec1, vec2):
    return vec1*vec2.dot(vec1)/vec1.dot(vec1)

def split(wall, speed):
    parallel = projects(wall, speed)
    perpendicular = speed - parallel
    return parallel, perpendicular
