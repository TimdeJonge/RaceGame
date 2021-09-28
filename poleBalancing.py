#%%
import gym
import pandas as pd
import time 
from neuralNet import Population

'''
Reset: Initial state
Step: current state, reward, done , info
Render: Opens window
close: Closes window
'''
PLAYER_AMOUNT = 30

class Game:
    def __init__(self, env, verbose, visualise):
        self.generation = 0
        self.population = Population(PLAYER_AMOUNT)
        self.population.create_population(4,1)
        self.innovation_df = pd.DataFrame(columns = ['Abbrev', 'Innovation_number'])
        self.total_nodes = 6
        self.environment = gym.make(env)
        self.verbose = verbose
        self.visualise = visualise
        self.max_fitness = 0

    def run(self):
        while self.max_fitness < 1000:
            self.calculate()
            self.reproduce()

    def calculate(self):
        network_Score = []
        num_steps = 5000
        
        for network in self.population:
            obs = self.environment.reset()
            fitness = 0
            for _ in range(num_steps):
                network.run_live(obs)
                action = 0 if network.state[4] < 0.5 else 1
                obs, rewards, done, info = self.environment.step(action)
                fitness += rewards

                if self.visualise:
                    self.environment.render()
                    time.sleep(0.001)
                
                if done:
                    network.fitness = fitness
                    network_Score.append(network.fitness)
                    break
        self.max_fitness = max(network_Score)
        if self.visualise:
            self.environment.close()
        if self.verbose:
            print(f'Achieved scores: {network_Score}')
            print(f'Maximal fitness {self.max_fitness}')
            
    def reproduce(self):
        self.population.advance_generation()

#%%

game = Game(env = 'CartPole-v1', verbose = True, visualise = True)
game.run()

# Number of steps you run the agent for 

# %%
