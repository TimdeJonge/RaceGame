import copy

file = open("baby_park.txt", "r")
array = []
for line in file:
    array.append([])
    line = line.split(' ')
    for element in line:
        if element == "1":
            array[-1].append(300)
        elif element == "0":
            array[-1].append(0)
        else:
            pass
        # pass because there is a \n element at the end of the line
        # if not for this "else" it crashes the whole thing
file.close()
# Let's call the following "testing":

for i in range(len(array)):
    if array[i][0] != 300:
        print("STRUGGLES at i = ", i)
    if array[i][-1] != 300:
        print("struggles at i = ", i)

for j in range(len(array[0])):
    if array[0][j] != 300:
        print("STRUGGLES at j = ", j)

    if array[-1][j] != 300:
        print("struggles at j = ", j)


def neighbours(input_array, y, x):
    return input_array[y-1][x], input_array[y+1][x], input_array[y][x-1], input_array[y][x+1]
stop = False
while not stop:
    stop = True
    array2 = copy.deepcopy(array)
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array2[i][j] == 0:
                stop = False
                maximum = max(neighbours(array2, i, j))
                if maximum != 0:
                    array[i][j] = maximum - 1

file2 = open("baby_park2.txt", "w")
for row in array:
    for element in row:
        file2.write(str(element) + " ")
    file2.write("\n")
file2.close()
