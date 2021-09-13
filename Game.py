import numpy as np
from Player import Player
from Global import BLACK, SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_AMOUNT
from Global import WHITE, BACKGROUND_COLOUR, FRAME_RATE, RED, GREEN, GENERATION_TIME
from neuralNet import AI_network
import pandas as pd
from levels import  level, create_obstacles, turny, turny_checkpoints, richard, level_checkpoints
from copy import deepcopy
import pygame


class Game(object):
    def __init__(self):
        self.debug = True
        self.level = turny
        self.obstacle_list = create_obstacles(self.level)
        self.checkpoints = turny_checkpoints
        self.counter = 0
        self.generation = 0
        self.sounds = self.init_sounds()
        self.player_list = [Player() for _ in range(PLAYER_AMOUNT)]
        self.network_list = [AI_network() for _ in range(PLAYER_AMOUNT)]
        self.innovation_df = pd.DataFrame(columns = ['Abbrev', 'Innovation_number'])
        self.total_nodes = 9
        for network in self.network_list:
            network.create_network(6,2,self.innovation_df)
            for _ in range(10):
                self.innovation_df, self.total_nodes = network.procreate(self.innovation_df, self.total_nodes)
            network.build(self.total_nodes)
        self.player = self.player_list[0]
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
                    pass
                    #self.player.restart(np.array([300, 300]), np.array([1.0, 1.0]))
                elif event.key == pygame.K_SPACE:
                    self.player.speed_boost = True
                elif event.key == pygame.K_d:
                    self.debug = not self.debug
                elif event.key == pygame.K_f:
                    self.reproduce()
                elif event.key == pygame.K_p:
                    self.player_list[-1].human = True
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
        if not self.counter%(GENERATION_TIME*FRAME_RATE):
            self.reproduce()
            #self.food.radius = max(30, self.food.radius)
        for i in range(len(self.player_list)):
            if not self.player_list[i].human:
                distances = self.player_list[i].observe(self.obstacle_list, self.level) 
                network = self.network_list[i]   
                network.run_live(distances)
                if network.state[6] < -0.2: 
                    self.player_list[i].speed_down = True 
                    self.player_list[i].speed_up = False
                elif network.state[6] > 0:
                    self.player_list[i].speed_down = False
                    self.player_list[i].speed_up = True
                if network.state[7] > 0.1:
                    self.player_list[i].turn = 'right'
                elif network.state[7] < -0.1:
                    self.player_list[i].turn = 'left'
                else: 
                    self.player_list[i].turn = 'neutral'
            self.player_list[i].update(self.obstacle_list, self.level, self.checkpoints, self.counter)
        # The camera is a bit of magic in how it works. Don't mess with it too much and all will be fine.
        # Just subtract self.camera from everything that needs to be drawn on screen and it will work.
        self.camera = (self.camera*(19) +
                       self.player.position -
                       np.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2]) +
                       self.player.speed*(40))*(.05)

    
    def reproduce(self):
        self.generation += 1
        if self.generation%2:
            self.level = turny
            self.obstacle_list = create_obstacles(self.level)
            self.checkpoints = turny_checkpoints
        else:
            self.level = level
            self.obstacle_list = create_obstacles(self.level)
            self.checkpoints = level_checkpoints
        for i in range(len(self.player_list)):
            self.network_list[i].fitness = (self.player_list[i].fitness + self.network_list[i].fitness)/2
            self.network_list[i].last_checkpoint = self.player_list[i].last_checkpoint
        self.network_list.sort(key = lambda x: (-x.fitness, x.last_checkpoint))
        del self.network_list[10:]
        #print(self.network_list[0].weights)
        for i in range(len(self.network_list)):
            self.network_list[i].fitness = 0
            for _ in range(4):
                fitwork = deepcopy(self.network_list[i])
                self.innovation_df, self.total_nodes = fitwork.procreate(self.innovation_df, self.total_nodes)
                self.network_list.append(fitwork)
        self.player_list = [Player() for _ in range(30)]
        self.player = self.player_list[0]
        self.player_list[0].colour = RED
        self.player_list[1].colour = GREEN

    def draw_debug(self, screen):
        """Draws debug strings. Contents can be varied in the initial string.
        :param screen      the screen on which to draw debug""" 
        font = pygame.font.SysFont('Console', 20, False, False)
        pygame.draw.circle(screen, BLACK, [600, 400], 0)
        debug_string1 = "distances = " + str(self.player.observe(self.obstacle_list, self.level))
        debug_string2 = "aim = " + str(self.player.turn)
        debug_string3 = "Total speed = " + str(np.linalg.norm(self.player.speed))
        debug_string4 = "Location = " + str(self.player.position)
        text1 = font.render(debug_string1, True, WHITE)
        text2 = font.render(debug_string2, True, WHITE)    
        text3 = font.render(debug_string3, True, WHITE)
        text4 = font.render(debug_string4, True, WHITE)
        screen.blit(text1, [0, 0])
        screen.blit(text2, [0, 15])
        screen.blit(text3, [0, 30])
        screen.blit(text4, [0, 45])
        for checkpoint in self.checkpoints:
            pygame.draw.line(screen, BLACK, np.array(checkpoint[0])- self.camera,
                         np.array(checkpoint[1])- self.camera)

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
        total_seconds = GENERATION_TIME-(counter // FRAME_RATE)%GENERATION_TIME 
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        output_string = "Time: {0:02}:{1:02}".format(minutes, seconds)
        text = font.render(output_string, True, WHITE)
        screen.blit(text, [SCREEN_WIDTH - 150, SCREEN_HEIGHT - 20])

    @staticmethod
    def init_sounds():
        HIT_SOUND = pygame.mixer.Sound("hit.ogg")
        return HIT_SOUND
