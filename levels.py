from Polygon import Polygon
from Obstacle import Obstacle
from Global import BLOCK_SIZE
import numpy as np

level = [[1 for _ in range(6)] for _ in range(8)]
for i in range(3):
    level[i+2][1] = 0
    level[1][i+1] = 0
    level[i+1][3] = 0
    level[i+3][4] = 0
    level[6][i+2] = 0
obstacle_list = [[1 for _ in range(6)] for _ in range(8)]
for i in range(8):
    for j in range(6):
        if level[i][j] == 1:
            obstacle_list[i][j] = Polygon([(i*BLOCK_SIZE, j*BLOCK_SIZE),
                              ((i+1)*BLOCK_SIZE, (j)*BLOCK_SIZE),
                              ((i+1)*BLOCK_SIZE, (j+1)*BLOCK_SIZE),
                              ((i)*BLOCK_SIZE, (j+1)*BLOCK_SIZE)])
        else:
            obstacle_list[i][j] = Obstacle()

level[5][1] = 1
obstacle_list[5][1] = Polygon([(5*BLOCK_SIZE, 1*BLOCK_SIZE),
                              (6*BLOCK_SIZE, 1*BLOCK_SIZE),
                              (6*BLOCK_SIZE, 2*BLOCK_SIZE)])
level[6][2] = 1
obstacle_list[6][2] = Polygon([(6*BLOCK_SIZE, 2*BLOCK_SIZE),
                              (7*BLOCK_SIZE, 2*BLOCK_SIZE),
                              (7*BLOCK_SIZE, 3*BLOCK_SIZE)])
level[5][2] = 1
obstacle_list[5][2] = Polygon([(5*BLOCK_SIZE, 2*BLOCK_SIZE),
                              (6*BLOCK_SIZE, 3*BLOCK_SIZE),
                              (5*BLOCK_SIZE, 3*BLOCK_SIZE)])

obstacle_list[3][5] = Polygon([(3*BLOCK_SIZE, 5*BLOCK_SIZE),
                              (4*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                              (4*BLOCK_SIZE, 6*BLOCK_SIZE),
                              (3*BLOCK_SIZE, 6*BLOCK_SIZE)])
obstacle_list[4][5] = Polygon([(4*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                          (5*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                          (5*BLOCK_SIZE, 6*BLOCK_SIZE),
                          (4*BLOCK_SIZE, 6*BLOCK_SIZE)])
obstacle_list[5][5] = Polygon([(5*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                      (6*BLOCK_SIZE, 5*BLOCK_SIZE),
                      (6*BLOCK_SIZE, 6*BLOCK_SIZE),
                      (5*BLOCK_SIZE, 6*BLOCK_SIZE)])
    
level = np.ones([6,8])
for i in range(3):
    level[1,i+2] = 0
    level[i+1,1] = 0
    level[3,i+1] = 0
    level[4,i+3] = 0
    level[i+2,6] = 0
obstacle_list = np.zeroes([6,8])
for x in range(8):
    for y in range(6):
        if level[x,y] == 1:
            obstacle_list[x,y] = Polygon([(x*BLOCK_SIZE, y*BLOCK_SIZE),
                              ((x+1)*BLOCK_SIZE, (y)*BLOCK_SIZE),
                              ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE),
                              ((x)*BLOCK_SIZE, (y+1)*BLOCK_SIZE)])
        else:
            obstacle_list[x,y] = Obstacle()
            
            
            
print(level)