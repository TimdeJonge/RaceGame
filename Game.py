import numpy as np
from Player import Player
from Global import BLOCK_SIZE, BLACK, PLAYER_COLOUR, SCREEN_WIDTH, SCREEN_HEIGHT
from Global import WHITE, BACKGROUND_COLOUR, FRAME_RATE, BLUE, GREEN
from Network import Network
from Food import Food
from levels import box, level, create_obstacles
from copy import deepcopy
import pygame


class Game(object):
    def __init__(self):
        self.debug = False
        self.level = level
        self.obstacle_list = create_obstacles(self.level)
        self.checkpoints = [[(BLOCK_SIZE, BLOCK_SIZE), (2*BLOCK_SIZE, 2*BLOCK_SIZE)],
                      [(5*BLOCK_SIZE, BLOCK_SIZE),(5*BLOCK_SIZE, 2*BLOCK_SIZE)],
                       [(6*BLOCK_SIZE, 4*BLOCK_SIZE),(7*BLOCK_SIZE, 5*BLOCK_SIZE)],
                       [(3*BLOCK_SIZE,4*BLOCK_SIZE), (4*BLOCK_SIZE,3*BLOCK_SIZE)]]
        self.counter = 0
        self.sounds = self.init_sounds()
        self.player_list = [Player(np.array([4.5*BLOCK_SIZE, 1.6*BLOCK_SIZE]), 
                                   np.array([1.0, 0.05]), 5, BLACK, 
                                   Network([6,4,3])) for i in range(50)]
        #self.player_list[0] = Player(np.array([4.5*BLOCK_SIZE, 1.6*BLOCK_SIZE]), np.array([1.0,0.01]), 5, PLAYER_COLOUR)
        self.player = self.player_list[0]
        self.turn = "neutral"
        self.acceleration = False
        self.brake = False
        self.camera = self.player.position - np.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2])
        #self.food = Food(800, 800, 1400, 1000)

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
                    self.player.restart(np.array([300, 300]), np.array([1.0, 1.0]))
                elif event.key == pygame.K_SPACE:
                    self.player.speed_boost = True
                elif event.key == pygame.K_f:
                    self.reproduce(1)
                elif event.key == pygame.K_p:
                    self.player_list[-1].network = "human"
                    self.player = self.player_list[-1]
                    self.player_list[-1].colour = WHITE

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
        if not self.counter%(90*FRAME_RATE):
            generation = self.counter // (90*FRAME_RATE)
# =============================================================================
#             if generation%2:
#                 self.food = Food(200,200,600,800)
#             else:
#                 self.food = Food(800, 800, 1400, 1000)
# =============================================================================
            self.reproduce(generation)
            #self.food.radius = max(30, self.food.radius)
        for player in self.player_list:
            player.update(self.obstacle_list, self.level, self.checkpoints)
        # The camera is a bit of magic in how it works. Don't mess with it too much and all will be fine.
        # Just subtract self.camera from everything that needs to be drawn on screen and it will work.
# =============================================================================
#         self.camera = (self.camera*(19) +
#                        self.player.position -
#                        np.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2]) +
#                        self.player.speed*(40))*(.05)
# 
# =============================================================================
        self.camera =np.array([200,200])
        
    def get_fitness(self, player):
        return player.fitness
    
    def reproduce(self, generation):
        print("THEY FUUUUUUUUUCKED")
        self.player_list.sort(key = self.get_fitness, reverse = True)
        self.player = self.player_list[0]
        del self.player_list[5:]
        for player in self.player_list:
            player.checkpoint = 2
            player.fitness = 0
            player.position = np.array([4.5*BLOCK_SIZE, 1.6*BLOCK_SIZE])
            player.speed = np.array([1.0, 0.01])
        for i in range(len(self.player_list)):
            if self.player_list[i].network != "human":
                for _ in range(8-i):
                    fitwork = deepcopy(self.player_list[i].network)
                    if generation > 3: 
                        generation = 3
                    fitwork.procreate(.6/(generation+1),.3/(generation+1))

                    self.player_list.append(Player(np.array([4.5*BLOCK_SIZE, 1.6*BLOCK_SIZE]),
                                       np.array([1.0, 0.01]), 5, BLACK, 
                                       fitwork))
        self.player_list[0].colour = BLUE
        self.player_list[1].colour = GREEN
        
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
        pygame.draw.line(screen, BLACK, np.array(self.checkpoints[self.player.checkpoint][0]) - self.camera,
                         np.array(self.checkpoints[self.player.checkpoint][1]) - self.camera)

    def draw_screen(self, screen):
        """Calls all draw methods.
        :param screen       the screen on which to draw all objects"""
        screen.fill(BACKGROUND_COLOUR)
        for obstacle_row in self.obstacle_list:
            for obstacle in obstacle_row:
                obstacle.draw(screen, self.camera)
        for player in self.player_list:
            player.draw_player(screen, self.camera)
        self.draw_time(screen, self.counter)
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
        total_seconds = 90-(counter // FRAME_RATE)%90 
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        output_string = "Time: {0:02}:{1:02}".format(minutes, seconds)
        text = font.render(output_string, True, WHITE)
        screen.blit(text, [SCREEN_WIDTH - 150, SCREEN_HEIGHT - 20])

    @staticmethod
    def init_sounds():
        HIT_SOUND = pygame.mixer.Sound("hit.ogg")
        return HIT_SOUND
