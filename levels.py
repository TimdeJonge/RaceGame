from Polygon import Polygon
from Obstacle import Obstacle
from Global import BLOCK_SIZE

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
level_checkpoints = [[(BLOCK_SIZE, BLOCK_SIZE), (2*BLOCK_SIZE, 2*BLOCK_SIZE)],
                   [(5*BLOCK_SIZE, BLOCK_SIZE),(5*BLOCK_SIZE, 2*BLOCK_SIZE)],
                   [(6*BLOCK_SIZE, 4*BLOCK_SIZE),(7*BLOCK_SIZE, 5*BLOCK_SIZE)],
                   [(3*BLOCK_SIZE,4*BLOCK_SIZE), (4*BLOCK_SIZE,3*BLOCK_SIZE)]]
    

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
            elif level[y][x] == 4: 
                obstacle_list[y][x] = Polygon([((x+1)*BLOCK_SIZE, y*BLOCK_SIZE),
                                     ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE),
                                     (x*BLOCK_SIZE, (y+1)*BLOCK_SIZE)])
            elif level[y][x] == 5: 
                 obstacle_list[y][x] = Polygon([((x+1)*BLOCK_SIZE, y*BLOCK_SIZE),
                                     (x*BLOCK_SIZE, y*BLOCK_SIZE),
                                     (x*BLOCK_SIZE, (y+1)*BLOCK_SIZE)])
            else:
                obstacle_list[y][x] = Obstacle()
    return(obstacle_list)
    
turny = [[1 for x in range(9)] for y in range(9)]
turny[2][1] = 0
turny[3][2] = 2
turny[3][1] = 3
turny[4][2] = 0
turny[5][2] = 4
turny[5][1] = 5
turny[6][1] = 0
turny[7][1] = 3
turny[7][2] = 0 
turny[7][3] = 0 
turny[7][4] = 0
turny[6][4] = 0 
turny[5][4] = 0 
turny[4][4] = 5
turny[4][5] = 4
turny[3][5] = 5
turny[3][6] = 4
turny[2][6] = 0
for i in range(6):
    turny[1][i+1] = 0
height = 14
width = 11
turny_checkpoints = [[(BLOCK_SIZE, BLOCK_SIZE), (2*BLOCK_SIZE, 2*BLOCK_SIZE)],
                             [(6*BLOCK_SIZE, 2*BLOCK_SIZE), (7*BLOCK_SIZE, BLOCK_SIZE)],
                             [(4*BLOCK_SIZE, 5*BLOCK_SIZE), (5*BLOCK_SIZE, 5*BLOCK_SIZE)],
                             [(3*BLOCK_SIZE, 7*BLOCK_SIZE), (3*BLOCK_SIZE, 8*BLOCK_SIZE)], 
                             [(2*BLOCK_SIZE, 4*BLOCK_SIZE), (3*BLOCK_SIZE, 4*BLOCK_SIZE)]]


richard = [[1 for x in range(width)] for y in range(height)]
# Y - X 
for i in range(1,4):
    richard[1][i+2] = 0
    richard[9][i] = 0
    richard[9][i+4] = 0
    richard[9+i][7] = 0
    richard[9+i][1] = 0
    richard[11][i+2] = 0
    richard[12][i] = 0
    richard[12][i+4] = 0

    
for i in range(1,8):
    richard[i+1][5] = 0 
    richard[i+1][3] = 0

