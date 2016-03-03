from levels import baby_park
from Vector import Vector
import time

obstacle_list = baby_park
file = open("baby_park.txt", "w")
bounding_box = [(0, 0), (10, 10)]
fixed = dict()
now = time.clock()
for i in range(1, bounding_box[1][0] - 1):
    for j in range(1, bounding_box[1][1] - 1):
        fixed_bool = False
        for obstacle in obstacle_list:
            if obstacle.contains(Vector((i+bounding_box[0][0], j + bounding_box[0][1]))):
                fixed_bool = True
                break
        if fixed_bool:
            fixed[(i, j)] = True
        else:
            fixed[(i, j)] = False


for i in range(bounding_box[1][1]):
    fixed[(0, i)] = True
    fixed[(bounding_box[1][0] - 1, i)] = True

for i in range(bounding_box[1][0]):
    fixed[(i, 0)] = True
    fixed[(i, bounding_box[1][1] - 1)] = True

print(time.clock() - now)
array = [[fixed[(i, j)] for i in range(bounding_box[1][0])] for j in range(bounding_box[1][1])]
print(time.clock() - now)
for row in array:
    for element in row:
        if element:
            file.write("1 ")
        else:
            file.write("0 ")
    file.write("\n")
print(time.clock() - now)

# TODO: Generate the actual heat map.
# Do this by using the above dict to populate an array.
# Everything that's fixed initially gets height 1000.
# Everything that has a neighbour that's fixed gets 2 less than the value of the neighbour.
# Incidentally, this is equal to the value assigned last pass minus 2.
# Save the result to a file or something.
