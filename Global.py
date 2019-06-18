import math

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BROWN = (150, 75, 0)

BACKGROUND_COLOUR = (64, 164, 223)
PLAYER_COLOUR = (255, 236, 230)
OBSTACLE_COLOUR = (194, 178, 128)

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FRAME_RATE = 60
BLOCK_SIZE = 100


ACCELERATION_DEFAULT = 0.05

sin_values = [math.sin(math.pi*i/180) for i in range(360)]
def sin(x):
    phi = int(x)%360
    return sin_values[phi]

def cos(x):
    phi = int(x)%360
    return sin_values[90-phi]
