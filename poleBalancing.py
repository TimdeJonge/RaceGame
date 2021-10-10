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
PLAYER_AMOUNT = 30
# Game options: 
# 'CartPole-v1' 
# 'MountainCar-v0'
# 'MountainCarContinuous-v0'
# BipedalWalker-v3

class Game:
    def __init__(self, env, verbose, visualise):
        self.generation = 0
        self.population = Population(PLAYER_AMOUNT)
        self.environment = gym.make(env)
        input_nodes = self.environment.observation_space.shape[0]
        output_nodes = self.environment.action_space.shape[0]
        init_mutations = 3
        self.output_node_number = input_nodes + output_nodes - 1
        self.population.create_population(input_nodes,output_nodes, init_mutations)
        self.innovation_df = pd.DataFrame(columns = ['Abbrev', 'Innovation_number'])
        self.total_nodes = input_nodes + output_nodes + 1
        self.verbose = verbose
        self.visualise = visualise
        self.max_fitness = 0

    def run(self):
        while self.max_fitness < self.environment._max_episode_steps:
            self.calculate()
            self.reproduce()
        
    def calculate(self):
        network_Score = []
        num_steps = self.population.generation * 100
        num_steps = max(num_steps, 10000)
        for _, network in enumerate(self.population):
            obs = self.environment.reset()
            for _ in range(num_steps):
                network.run_live(obs)
                action = []
                for output_nodes in range(self.environment.action_space.shape[0]):
                    output_node = self.environment.observation_space.shape[0] + output_nodes
                    if network.state[output_node] < 0.3:
                        action.append(1)
                    elif network.state[output_node] > 0.7:
                        action.append(-1)
                    else:
                        action.append(0)

                obs, rewards, done, info = self.environment.step(action)
                if (self.visualise):
                    self.environment.render()
                    time.sleep(0.001)
                if done:
                    network.fitness = rewards
                    network_Score.append(network.fitness)
                    if self.verbose:
                        print(f'Fitness: {network.fitness}')
                    break
        self.max_fitness = max(network_Score)
        if self.visualise:
            self.environment.close()
        print(f'Maximal fitness of generation: {self.max_fitness}')
            
    def reproduce(self):
        self.population.advance_generation(reduce_species=False)
        print(f'Generation: {self.population.generation}')
        
#%%

game = Game(env = 'BipedalWalker-v3', verbose = False, visualise = True)
game.run()

# Number of steps you run the agent for 

# %%
