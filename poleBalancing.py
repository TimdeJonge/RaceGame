#%%
import gym
import pandas as pd
from NeuralNet.neuralNet import Population
import logging
import random 
import time
import os
import sys


# Game options: 
# 'CartPole-v1' 
# 'MountainCar-v0'
# 'MountainCarContinuous-v0'
# BipedalWalker-v3, 500
# Acrobot-v1    
# LunarLanderContinuous-v2, 150

# TODO:
#   - Scale fitness with initial velocity
#   - Clean up logging coming from neuralNet 
#   - Network fitness from multiple runs for random push
#   - Clean up replay saving

def main():
    for init_connections in [16]:
        for trial in [1]:
            game = Game(
                        pop_size=50,
                        name = f'init_{init_connections}_{trial}',
                        log_run=True,
                        init_connections=init_connections
                        )

            while (game.level_number < 25) and (game.population.generation < 150):
                if game.level_number == 0 and game.population.generation < 15:
                    fitness_weights = {}
                else:
                    fitness_weights = {'life_span' : 0}
                options={'push' : [random.choice([-1,1])*game.level_number*100, game.level_number*-50], 'fitness_weights' : fitness_weights}
                game.calculate(options=options)
                game.reproduce()
                game.write_out(f"replays/{game.name}.csv")
            else:
                for species_key in game.population.species_dict:
                    game.population.species_dict[species_key][0].dump_connections(f'replays/{game.name}/connections_{species_key}.csv')



def replay(file_location):
    Game(pop_size=1).environment.replay(file_location)

class Game:
    def __init__(self, pop_size, log_run=False, name=time.strftime('%y%m%d-%H%M'), init_connections='all'):
        self.population = Population(pop_size)
        self.environment = gym.make('LunarLanderContinuous-v2')
        self.input_nodes = self.environment.observation_space.shape[0] 
        try:
            self.output_nodes = self.environment.action_space.shape[0] # GAME DEPENDENT
            self.continuous = True
        except IndexError:
            self.output_nodes = 1
            self.continuous = False   
        self.population.create_population(self.input_nodes,self.output_nodes, init_connections=init_connections)
        self.log = pd.DataFrame() if log_run else None
        self.level_number = 0
        self.level_up = False
        write_folder = f'./replays/{name}'
        if not os.path.exists(write_folder):
            os.mkdir(write_folder)
        self.name = name
        self.logger = self._init_logger(log_run) 


    def _init_logger(self, log_run):
        logger = logging.getLogger(__name__)
        logger.handlers = []
        if log_run:
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
            fh = logging.FileHandler(f'replays/{self.name}.log')
            formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s', datefmt='%H:%M:%S')
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(ch)
            logger.addHandler(fh)
        else:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
        return logger


    def actions(self, state):
        action = []
        for output_node in range(self.output_nodes):
            output_node_index = self.input_nodes + output_node
            if self.continuous:
                action.append(state[output_node_index]*2-1)
            elif state[output_node_index] < 0.3:
                action.append(1)
            elif state[output_node_index] > 0.7:
                action.append(-1)
            else:
                action.append(0)
        return action[0] if self.output_nodes == 1 else action

    def calculate(self, options):
        self.logger.info(f'Level {self.level_number}')
        for key, value in options.items():
            self.logger.info(f'{key} : {value}')

        num_steps = 500
        for i, network in enumerate(self.population):
            solved=False
            obs = self.environment.reset(seed=self.level_number, options=options)
            fitness = []
            for _ in range(num_steps):
                state = network.run_live(obs)
                action = self.actions(state)
                obs, rewards, done, info = self.environment.step(action)
                fitness.append(rewards)
                solved = info.get('success', False)
                if done:
                    break
            network.fitness = max(sum(fitness), 0.001)
            if (i == 0) and (self.population.generation % 5 == 0):
                self.environment.log(f"replays/{self.name}/sample_{self.population.generation}.txt")
            if solved:             
                self.environment.log(f"replays/{self.name}/gen_{self.population.generation}_{str(network.id)}.txt")
                self.level_up = True
        if self.level_up:
            self.level_number += 1
            self.level_up = False

    def reproduce(self):
        if self.log is not None:
            self.log = self.log.append(self.population.log().assign(level_number=self.level_number))
        species_log_dict = self.population.advance_generation()
        species_log = str(pd.DataFrame.from_dict(species_log_dict, orient='index'))
        self.logger.debug('\n' + species_log)
        self.logger.info(f'Generation: {self.population.generation}')
    
    def write_out(self, location):
        if self.log is not None:
            self.log.to_csv(location)

# %%
if __name__ == '__main__':
    main()




