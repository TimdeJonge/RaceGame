import numpy as np
from Player import Player
from Polygon import Polygon
from Obstacle import Obstacle
from Global import BLOCK_SIZE, BLACK, PLAYER_COLOUR, SCREEN_WIDTH, SCREEN_HEIGHT
from Global import WHITE, BACKGROUND_COLOUR, FRAME_RATE
import math
import pygame


class Game(object):
    def __init__(self):
        #TODO: FIX I,J
        self.debug = True
        self.level = [[1 for _ in range(6)] for _ in range(8)]
        for i in range(8):
            for j in range(6):
                self.level[i][j] = 1
        for i in range(3):
            self.level[i+2][1] = 0
            self.level[1][i+1] = 0
            self.level[i+1][3] = 0
            self.level[i+3][4] = 0
            self.level[6][i+2] = 0
        self.level_width = len(self.level[0])
        self.level_height = len(self.level)
        self.obstacle_list = [[1 for _ in range(6)] for _ in range(8)]
        for i in range(8):
            for j in range(6):
                if self.level[i][j] == 1:
                    self.obstacle_list[i][j] = Polygon([(i*BLOCK_SIZE, j*BLOCK_SIZE),
                                      ((i+1)*BLOCK_SIZE, (j)*BLOCK_SIZE),
                                      ((i+1)*BLOCK_SIZE, (j+1)*BLOCK_SIZE),
                                      ((i)*BLOCK_SIZE, (j+1)*BLOCK_SIZE)])
                else:
                    self.obstacle_list[i][j] = Obstacle()

        self.level[5][1] = 1
        self.obstacle_list[5][1] = Polygon([(5*BLOCK_SIZE, 1*BLOCK_SIZE),
                                      (6*BLOCK_SIZE, 1*BLOCK_SIZE),
                                      (6*BLOCK_SIZE, 2*BLOCK_SIZE)])
        self.level[6][2] = 1
        self.obstacle_list[6][2] = Polygon([(6*BLOCK_SIZE, 2*BLOCK_SIZE),
                                      (7*BLOCK_SIZE, 2*BLOCK_SIZE),
                                      (7*BLOCK_SIZE, 3*BLOCK_SIZE)])
        self.level[5][2] = 1
        self.obstacle_list[5][2] = Polygon([(5*BLOCK_SIZE, 2*BLOCK_SIZE),
                                      (6*BLOCK_SIZE, 3*BLOCK_SIZE),
                                      (5*BLOCK_SIZE, 3*BLOCK_SIZE)])
    
        self.obstacle_list[3][5] = Polygon([(3*BLOCK_SIZE, 5*BLOCK_SIZE),
                                      (4*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                                      (4*BLOCK_SIZE, 6*BLOCK_SIZE),
                                      (3*BLOCK_SIZE, 6*BLOCK_SIZE)])
        self.obstacle_list[4][5] = Polygon([(4*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                                  (5*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                                  (5*BLOCK_SIZE, 6*BLOCK_SIZE),
                                  (4*BLOCK_SIZE, 6*BLOCK_SIZE)])
        self.obstacle_list[5][5] = Polygon([(5*BLOCK_SIZE, 5.1*BLOCK_SIZE),
                              (6*BLOCK_SIZE, 5*BLOCK_SIZE),
                              (6*BLOCK_SIZE, 6*BLOCK_SIZE),
                              (5*BLOCK_SIZE, 6*BLOCK_SIZE)])
        # RELEVANT VALUES
        self.counter = 0
        self.sounds = self.init_sounds()

        self.player_list = [Player(self.sounds, np.array([1.5*BLOCK_SIZE, 1.6*BLOCK_SIZE]), np.array([0.0, 1.0]), 5, BLACK) for i in range(1)]
        self.player_list[0] = Player(self.sounds, np.array([1.5*BLOCK_SIZE,1.5*BLOCK_SIZE]), np.array([1.0,0.0]), 5, PLAYER_COLOUR, True)
        self.player = self.player_list[0]

# =============================================================================
#         if self.level == "Baby Park":
#             self.obstacle_list = levels.baby_park
#         elif self.level == "Circles":
#             self.obstacle_list = levels.circles
#         elif self.level == "L":
#             self.obstacle_list = levels.l
#         elif self.level == "Clear":
#             self.obstacle_list = levels.clear
#         elif self.level == "Test":
#             self.obstacle_list = levels.test
#         elif self.level == "donut":
#             self.obstacle_list = levels.donut
#         elif self.level == "test_block":
#             self.obstacle_list = levels.test_block
# =============================================================================
        self.turn = "neutral"
        self.acceleration = False
        self.brake = False
        self.camera = self.player.position - np.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2])

    def process_events(self):
        """Handles all user inputs. Returns boolean "done" """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.player.turn = "left"
                elif event.key == pygame.K_RIGHT:
                    self.player.turn = "right"
                elif event.key == pygame.K_UP:
                    self.player.speed_up = True
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.player.speed_down = True
                elif event.key == pygame.K_r:
                    self.player.restart(np.array([100, 100]), np.array([1, 1]))
                elif event.key == pygame.K_SPACE:
                    self.player.speed_boost = True

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    self.player.turn = "neutral"
                elif event.key == pygame.K_UP:
                    self.player.speed_up = False
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.player.speed_down = False
        return False

    def update(self):
        self.counter += 1
        for player in self.player_list:
            player.update(self.obstacle_list, self.level)
            player.move(self.obstacle_list, self.level)
        # The camera is a bit of magic in how it works. Don't mess with it too much and all will be fine.
        # Just subtract self.camera from everything that needs to be drawn on screen and it will work.
        self.camera = (self.camera*(19) +
                       self.player.position -
                       np.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2]) +
                       self.player.speed*(40))*(.05)

    def draw_debug(self, screen):
        """Draws debug strings. Contents can be varied in the initial string.
        :param screen      the screen on which to draw debug""" 
        font = pygame.font.SysFont('Console', 20, False, False)
        pygame.draw.circle(screen, BLACK, [600, 400], 0)
        debug_string1 = "direction = " + str(self.player.direction)
        debug_string3 = "Total speed = " + str(np.linalg.norm(self.player.speed))
        debug_string4 = "Location = " + str(self.player.position)
        text1 = font.render(debug_string1, True, WHITE)
        text3 = font.render(debug_string3, True, WHITE)
        text4 = font.render(debug_string4, True, WHITE)
        screen.blit(text1, [0, 0])
        screen.blit(text3, [0, 30])
        screen.blit(text4, [0, 45])

    def draw_screen(self, screen):
        """Calls all draw methods.
        :param screen       the screen on which to draw all objects"""
        screen.fill(BACKGROUND_COLOUR)
        for obstacle_row in self.obstacle_list:
            for obstacle in obstacle_row:
                obstacle.draw(screen, self.camera)
        for player in self.player_list:
            player.draw_player(screen, self.camera)
        Game.draw_time(screen, self.counter)
        if self.debug:
            self.draw_debug(screen)

    @staticmethod
    def draw_time(screen, counter):
        """Draws time in the lower right corner
        Keyword arguments:
        :param screen       the screen on which to draw the time
        :param counter      the amount of frames past since the start of counting
        :type counter       int
        """
        font = pygame.font.SysFont('Console', 20, False, False)
        total_seconds = counter // FRAME_RATE
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        output_string = "Time: {0:02}:{1:02}".format(minutes, seconds)
        text = font.render(output_string, True, WHITE)
        screen.blit(text, [SCREEN_WIDTH - 150, SCREEN_HEIGHT - 20])

    @staticmethod
    def init_sounds():
        HIT_SOUND = pygame.mixer.Sound("hit.ogg")
        return HIT_SOUND
