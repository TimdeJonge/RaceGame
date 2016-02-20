from levels import baby_park
from Polygon import Polygon
from Circle import Circle
from Vector import Vector
import time

obstacle_list = baby_park

bounding_box = [(0, 0), (1200, 800)]
fixed = dict()
now = time.clock()
for i in range(1, bounding_box[1][0] - 1):
    for j in range(1, bounding_box[1][1] - 1):
        fixed_bool = False
        for obstacle in obstacle_list:
            if obstacle.collides_with(Vector((i+bounding_box[0][0], j + bounding_box[0][1]))):
                fixed_bool = True
                break
        if fixed_bool:
            fixed[(i,j)] = True
        else:
            fixed[(i,j)] = False


for i in range(bounding_box[1][1]):
    fixed[(0, i)] = True
    fixed[(bounding_box[1][0] - 1, i)] = True

for i in range(bounding_box[1][0]):
    fixed[(i, 0)] = True
    fixed[(i, bounding_box[1][1] - 1)] = True

# array = [[fixed[(i,j)] for i in range(bounding_box[1][0])] for j in range(bounding_box[1][1])]

print("Done")
print(time.clock() - now)

# TODO: Generate the actual heatmap.
# Do this by using the above dict to populate an array.
# Everything that's fixed initially gets height 1000.
# Everything that has a neighbour that's fixed gets 2 less than the value of the neighbour.
# Incidentally, this is equal to the value assigned last pass minus 2.
# Save the result to a file or something.
