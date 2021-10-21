from collections import defaultdict
import numpy as np
import pandas as pd

from pygame.constants import K_KP_PLUS
from time import localtime, strftime
import pygame

from Race.Player import Player
from Race.Global import BLACK, SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_AMOUNT
from Race.Global import WHITE, BACKGROUND_COLOUR, FRAME_RATE, RED, GREEN, GENERATION_TIME
from NeuralNet.neuralNet import Population
from Race.levels import  level, create_obstacles, turny, turny_checkpoints, richard, level_checkpoints, richard_checkpoints

class Game(object):
    def __init__(self):
        self.debug = True
        self.level = turny
        self.obstacle_list = create_obstacles(self.level)
        self.checkpoints = turny_checkpoints
        self.counter = 0
        self.generation = 0
        self.population = Population(PLAYER_AMOUNT)
        self.input_nodes = 6
        self.output_nodes = 2
        self.init_connections = 'all'
        self.population.create_population(self.input_nodes,self.output_nodes,self.init_connections)
        self.player_list = [Player(network=network) for network in self.population]
        self.player_active = 0
        self.innovation_df = pd.DataFrame(columns = ['Abbrev', 'Innovation_number'])
        self.total_nodes = 9
        self.last_checkpoint = 0
        self.max_fitness = 0 
        self.anger = 0
        self.display = True
        self.player = self.player_list[0]
        self.camera = self.player.position - np.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2])

    def process_events(self):
        """Handles all user inputs. Returns boolean "done" """
        for event in pygame.event.get():   
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if self.player.human:
                        self.player.turn = "left"

                elif event.key == pygame.K_RIGHT:
                    self.player.turn = "right"
                elif event.key == pygame.K_UP:
                    if self.player.human:
                        self.player.speed_up = True
                    else:
                        self.player_active = (self.player_active + 1) % len(self.population) 
                        self.player = self.player_list[self.player_active]
                elif event.key == pygame.K_DOWN:
                    if self.player.human:
                        self.player.speed_down = True
                    else:
                        self.player_active = (self.player_active - 1) %  len(self.population)  
                        self.player = self.player_list[self.player_active]
                elif event.key == pygame.K_r:
                    pass
                    #self.player.restart(np.array([300, 300]), np.array([1.0, 1.0]))
                elif event.key == pygame.K_SPACE:
                    self.player.speed_boost = True
                elif event.key == pygame.K_d:
                    self.debug = not self.debug
                elif event.key == pygame.K_q:
                    self.display = not self.display
                elif event.key == pygame.K_c:
                    print(self.player.network.connections.loc[self.player.network.connections['Enabled']])     
                elif event.key == pygame.K_n:
                    print(self.player.network.nodes)   
                elif event.key == pygame.K_o:
                    print(self.player.network.order)   
                elif event.key == pygame.K_f:
                    self.reproduce()
                    self.counter = 0
                elif event.key == pygame.K_t:
                    self.reproduce(keep_species=False)
                    self.counter = 0
                elif event.key == pygame.K_KP_PLUS:
                    self.counter -= 30*FRAME_RATE
                    self.last_checkpoint = self.counter
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
        if not self.counter%(GENERATION_TIME*FRAME_RATE) or (self.counter - self.last_checkpoint > FRAME_RATE*15):
            self.counter = 0
            self.reproduce()
        for player in self.player_list:
            if not player.human:
                distances = player.observe(self.obstacle_list, self.level) 
                player.network.run_live(distances)
                if player.network.state[6] < 0.4: 
                    player.speed_down = True 
                    player.speed_up = False
                elif player.network.state[6] > .5:
                    player.speed_down = False
                    player.speed_up = True
                if player.network.state[7] > 0.6:
                    player.turn = 'right'
                elif player.network.state[7] < 0.4:
                    player.turn = 'left'
                else: 
                    player.turn = 'neutral'
            checkpoint_hit = player.update(self.obstacle_list, self.level, self.checkpoints)
            if checkpoint_hit:
                self.last_checkpoint = self.counter
        # The camera is a bit of magic in how it works. Don't mess with it too much and all will be fine.
        # Just subtract self.camera from everything that needs to be drawn on screen and it will work.
        self.camera = (self.camera*(19) +
                       self.player.position -
                       np.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2]) +
                       self.player.speed*(40))*(.05)
    
    def reproduce(self, keep_species=True):
        self.generation += 1
        print(f'{strftime("%H:%M:%S", localtime())}: Starting Generation {self.generation}')
        if self.generation%10:
            self.level = richard
            self.obstacle_list = create_obstacles(self.level)
            self.checkpoints = richard_checkpoints
        elif self.generation%9:
            self.level = turny
            self.obstacle_list = create_obstacles(self.level)
            self.checkpoints = turny_checkpoints
        else:
            self.level = level
            self.obstacle_list = create_obstacles(self.level)
            self.checkpoints = level_checkpoints
        for player in self.player_list:
            player.network.fitness, player.fitness = player.fitness, (player.fitness + player.network.fitness)/2
        fitwork = max(self.population, key=lambda x: x.fitness)
        print(fitwork.fitness, fitwork.connections)
        if fitwork.fitness > self.max_fitness:
            self.max_fitness = fitwork.fitness
            self.anger = 0
        else:
            self.anger += 1
        if self.anger < 10 and keep_species:
            self.population.advance_generation()
        else:
            self.population.advance_generation(reduce_species=True)
        self.player_list = [Player(network=network) for network in self.population]
        self.player_active = 0
        self.player = self.player_list[self.player_active]
        for player in self.player_list:
            player.colour = colourpicker[player.network.species]
        # self.player_list[0].colour = RED
        # self.player_list[1].colour = GREEN

    def draw_debug(self, screen):
        """Draws debug strings. Contents can be varied in the initial string.
        :param screen      the screen on which to draw debug""" 
        debug_strings = [f"Following: Player {self.player_active}"]
        font = pygame.font.SysFont('Console', 20, False, False)
        if self.player.human:
            debug_strings[0] = "Following: You!"
        else:
            debug_strings[0] = f"Following: Player {self.player_active}"
        debug_strings.append(f"Fitness = {self.player.fitness}")
        debug_strings.append(f'Last Fitness = {self.player.network.fitness}')
        debug_strings.append(f'Species = {self.player.network.species}')
        debug_strings.append(f'Gas Pedal / Steering Wheel: {self.player.network.state[6:8]}')
        for i in range(len(debug_strings)):
            screen.blit(font.render(debug_strings[i], True, WHITE), [0, 0+20*i])
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
        for player in self.player_list[::-1]:
            player.draw_player(screen, self.camera)
        self.draw_time(screen, self.counter)
        font = pygame.font.SysFont('Console', 20, False, False)
        screen.blit(font.render(f"Generation {self.generation}", True, WHITE), (0, SCREEN_HEIGHT-20))
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


colourpicker = defaultdict(lambda : (np.random.randint(0,255, size=3)))