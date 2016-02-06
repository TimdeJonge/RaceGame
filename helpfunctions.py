from math import sqrt


def distance_point_line(point, lineA, lineB):
    """Given a point A and a line BC, calculates the distance from A to BC. (all points given in 2-tuples)"""
    # http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
    # Should I get an implementation of Vector somewhere, this can be made much prettier
    # Until then, this has to do.
    numerator = ((lineB[0] - lineA[0])*(lineA[1] - point[1]) - (lineA[0] - point[0])*(lineB[1] - lineA[1]))
    denominator = sqrt((lineB[0]- lineA[0])**2 + (lineB[1]- lineA[1])**2)
    return numerator/denominator

