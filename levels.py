from Polygon import Polygon
from Obstacle import Obstacle
from Global import BLOCK_SIZE, WHITE, BLACK

level = [[1 for x in range(8)] for y in range(6)]
for i in range(3):
    level[1][i+2] = 0
    level[i+1][1] = 0
    level[3][i+1] = 0
    level[4][i+3] = 0
    level[i+2][6] = 0
level[1][5] = 2
level[2][6] = 2
level[2][5] = 3
obstacle_list = [[1 for x in range(8)] for y in range(6)]
for x in range(8):
    for y in range(6):
        if level[y][x] == 1:
            obstacle_list[y][x] = Polygon([(x*BLOCK_SIZE, y*BLOCK_SIZE),
                              ((x+1)*BLOCK_SIZE, y*BLOCK_SIZE),
                              ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE),
                              (x*BLOCK_SIZE, (y+1)*BLOCK_SIZE)])
        elif level[y][x] == 2:
            obstacle_list[y][x] = Polygon([(x*BLOCK_SIZE, y*BLOCK_SIZE),
                                 ((x+1)*BLOCK_SIZE, y*BLOCK_SIZE),
                                 ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE)])
        elif level[y][x] == 3: 
            obstacle_list[y][x] = Polygon([(x*BLOCK_SIZE, y*BLOCK_SIZE),
                                 (x*BLOCK_SIZE, (y+1)*BLOCK_SIZE),
                                 ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE)])
        else:
            obstacle_list[y][x] = Obstacle()
            
            
print(level)
obstacle_list[5][3] = Polygon([(3*BLOCK_SIZE, 5*BLOCK_SIZE),
                              (4*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                              (4*BLOCK_SIZE, 6*BLOCK_SIZE),
                              (3*BLOCK_SIZE, 6*BLOCK_SIZE)])
obstacle_list[5][4] = Polygon([(4*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                          (5*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                          (5*BLOCK_SIZE, 6*BLOCK_SIZE),
                          (4*BLOCK_SIZE, 6*BLOCK_SIZE)])
obstacle_list[5][5] = Polygon([(5*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                      (6*BLOCK_SIZE, 5*BLOCK_SIZE),
                      (6*BLOCK_SIZE, 6*BLOCK_SIZE),
                      (5*BLOCK_SIZE, 6*BLOCK_SIZE)])
    

def create_obstacles(level):
    obstacle_list = [[1 for x in range(len(level[0]))] for y in range(len(level))]
    for x in range(len(level[0])):
        for y in range(len(level)):
            if level[y][x] == 1:
                obstacle_list[y][x] = Polygon([(x*BLOCK_SIZE, y*BLOCK_SIZE),
                                  ((x+1)*BLOCK_SIZE, y*BLOCK_SIZE),
                                  ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE),
                                  (x*BLOCK_SIZE, (y+1)*BLOCK_SIZE)])
            elif level[y][x] == 2:
                obstacle_list[y][x] = Polygon([(x*BLOCK_SIZE, y*BLOCK_SIZE),
                                     ((x+1)*BLOCK_SIZE, y*BLOCK_SIZE),
                                     ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE)])
            elif level[y][x] == 3: 
                obstacle_list[y][x] = Polygon([(x*BLOCK_SIZE, y*BLOCK_SIZE),
                                     (x*BLOCK_SIZE, (y+1)*BLOCK_SIZE),
                                     ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE)])
            else:
                obstacle_list[y][x] = Obstacle()
    return(obstacle_list)
    
box = [[0 for x in range(6)] for y in range(6)]
for x in range(6):
    for y in range(6):
        if x == 0 or x == 5 or y == 0 or y == 5:
            box[y][x] = 1
obstacle_list = create_obstacles(box)
