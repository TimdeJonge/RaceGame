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

class Game:
    def __init__(self, env, verbose, visualise):
        self.generation = 0
        self.population = Population(PLAYER_AMOUNT)
        self.environment = gym.make(env)
        input_nodes = self.environment.observation_space.high.size
        output_nodes = 1 #self.environment.action_space
        init_mutations = 0
        self.output_node_number = input_nodes + output_nodes - 1
        self.population.create_population(input_nodes,output_nodes, init_mutations)
        self.innovation_df = pd.DataFrame(columns = ['Abbrev', 'Innovation_number'])
        self.total_nodes = input_nodes + output_nodes + 1
        self.verbose = verbose
        self.visualise = visualise
        self.max_fitness = 0

    def run(self):
        while self.max_fitness < self.environment._max_episode_steps:
            self.visualise = bool(self.population.generation % 25 == 0)
            self.calculate()
            self.reproduce()
        
    def calculate(self):
        network_Score = []
        num_steps = 50000
        for count, network in enumerate(self.population):
            if self.verbose and count == 0:
                print(network.connections)
            obs = self.environment.reset()
            position_score = 0
            fitness = 0
            for _ in range(num_steps):
                network.run_live(obs)
                if network.state[self.output_node_number] < 0.1:
                    action = 0
                elif network.state[self.output_node_number] > 0.9:
                    action = 2
                else:
                    action = 1
                # action = 0 if network.state[4] < 0.5 else 1
                # if network.state[4] < 0.5:
                #     action = [-1]
                # else:
                #     action = [1]
                obs, rewards, done, info = self.environment.step(action)
                if obs[0] > position_score:
                    position_score = obs[0]
                fitness += rewards + abs(obs[0]) 

                if self.visualise:
                    self.environment.render()
                    time.sleep(0.001)

                if done:
                    network.fitness = fitness + position_score
                    network_Score.append(network.fitness)
                    if self.verbose:
                        print(f'Fitness of the network: {fitness}')
                    break
        self.max_fitness = max(network_Score)
        if self.visualise:
            self.environment.close()
        if self.verbose:
            print(f'Maximal fitness {self.max_fitness}')
            
    def reproduce(self):
        self.population.advance_generation(reduce_species=True)
        print(f'Generation: {self.population.generation}')
#%%

game = Game(env = 'MountainCar-v0', verbose = True, visualise = True)
game.run()

# Number of steps you run the agent for 

# %%
