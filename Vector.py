import math


class Vector:
    def __init__(self, amount, values=()):
        if type(amount) == list and amount != []:
            self.values = []
            for value in amount:
                self.values.append(float(value))
            self.amount = len(amount)
        elif type(amount) == tuple and amount != ():
            self.values = []
            for value in amount:
                self.values.append(float(value))
            self.amount = len(amount)
        elif type(amount) == int and values == []:
            self.amount = amount
            self.values = [0.0 for _ in range(amount)]
        elif type(amount) == int and values != []:
            self.values = []
            for value in values:
                self.values.append(float(value))
            self.amount = amount

    def __str__(self):
        super().__str__()
        return str(self.values)

    def linear_combination(self, other, alpha, beta):
        result = []
        for i in range(self.amount):
            result.append(self.values[i]*alpha + other.values[i]*beta)
        return Vector(self.amount, result)

    def scalar(self, alpha):
        return self.linear_combination(self, alpha, 0)

    def normalize(self):
        return self.scalar(1/self.norm())

    def inner(self, other):
        result = 0
        for i in range(self.amount):
            result += self.values[i]*other.values[i]
        return result

    def norm(self):
        return math.sqrt(self.inner(self))

    def projects(self, other):
        return self.scalar(other.inner(self)/self.inner(self))

    def distance_to_line(self, point_a, point_b):
        v = Vector(2, [point_b.values[1] - point_a.values[1], point_a.values[0] - point_b.values[0]])
        r = point_a - self
        return v.inner(r) / v.norm()

    def __sub__(self, other):
        return self.linear_combination(other, 1, -1)

    def __add__(self, other):
        return self.linear_combination(other, 1, 1)

    def to_int(self):
        int_values = []
        for value in self.values:
            int_values.append(int(value))
        return int_values

    def split(self, direction):
        parallel = (direction).projects(self)
        perpendicular = self - parallel
        return parallel, perpendicular
