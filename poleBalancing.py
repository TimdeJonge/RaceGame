#%%
import gym
from numpy.testing._private.utils import verbose
import pandas as pd
import time 
from neuralNet import Population
import numpy as np
'''
Reset: Initial state
Step: current state, reward, done , info
Render: Opens window
close: Closes window
'''
PLAYER_AMOUNT = 70


class Game:
    def __init__(self, env, verbose, visualise, logging):
        self.env = env
        self.generation = 0
        self.population = Population(PLAYER_AMOUNT)
        self.environment = gym.make(env)
        self.input_nodes = self.environment.observation_space.shape[0]
        try:
            self.output_nodes = self.environment.action_space.shape[0]
            self.Continuous = True
        except IndexError:
            self.output_nodes = 1
            self.Continuous = False   
        self.init_connections = 4
        self.output_node_number = self.input_nodes + self.output_nodes - 1
        self.population.create_population(self.input_nodes,self.output_nodes, self.init_connections)
        self.innovation_df = pd.DataFrame(columns = ['Abbrev', 'Innovation_number'])
        self.total_nodes = self.input_nodes + self.output_nodes + 1
        self.verbose = verbose
        self.visualise = visualise
        self.max_fitness = 0
        self.logging = logging
        self.log = pd.DataFrame()
        self.level_number = 0
    def run(self):
        while self.max_fitness < self.environment._max_episode_steps:
            self.calculate()
            self.reproduce()
        
    def calculate(self):
        network_Score = []
        num_steps = 500
        for num, network in enumerate(self.population):
            self.environment.seed(self.level_number)
            obs = self.environment.reset()
            fitness = []
            for _ in range(num_steps):
                network.run_live(obs)
                action = []
                
                for output_node in range(self.output_nodes):
                    output_node_index = self.input_nodes + output_node
                    if self.Continuous:
                        action.append(network.state[output_node_index]*2-1)
                    elif network.state[output_node_index] < 0.3:
                        action.append(1)
                    elif network.state[output_node_index] > 0.7:
                        action.append(-1)
                    else:
                        action.append(0)

                if self.output_nodes == 1:
                    obs, rewards, done, info = self.environment.step(action[0])
                else:
                    obs, rewards, done, info = self.environment.step(action)
                fitness.append(rewards + 1)
                if ((self.visualise) & (num == 0)  & 
                    (self.population.generation != 0)):
                    self.environment.render()
                    time.sleep(0.001)

                if done:
                    if self.verbose:
                        print(f'Fitness: {network.fitness}')
                    break

            if self.env == 'Acrobot-v1':
                network.fitness = np.mean(fitness)
            else:
                network.fitness = max(sum(fitness),0.001)
            network_Score.append(network.fitness)

        self.max_fitness = max(network_Score)
        network_Score.sort(reverse = True)
        if np.mean(network_Score[:4] ) > 200:
            self.level_number += 1
        if self.visualise:
            self.environment.close()
        print(f'Maximal fitness of generation: {self.max_fitness}')
            
    def reproduce(self):
        if self.logging:
            self.log = self.log.append(self.population.log())
            self.log.to_csv('test1.csv')
        self.population.advance_generation(reduce_species=False)
        print(f'Generation: {self.population.generation}')


#%%
# Game options: 
# 'CartPole-v1' 
# 'MountainCar-v0'
# 'MountainCarContinuous-v0'
# BipedalWalker-v3
# Acrobot-v1
#  LunarLanderContinuous-v2
game = Game(env = 'LunarLanderContinuous-v2' , verbose = False, visualise = True, logging = True)
game.run()

# Number of steps you run the agent for 

# %%
# Test 1: Enable disable off