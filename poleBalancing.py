#%%
import gym
import pandas as pd
from NeuralNet.neuralNet import Population
import logging

# Game options: 
# 'CartPole-v1' 
# 'MountainCar-v0'
# 'MountainCarContinuous-v0'
# BipedalWalker-v3, 500
# Acrobot-v1
#  LunarLanderContinuous-v2, 150

def main():
    game = Game(
                pop_size=70,
                logging_level = 'INFO'
                )
    while game.max_fitness < game.environment._max_episode_steps:
        print(game.level_number)
        if (game.level_number > 10) and (game.level_number < 20):
            fitness_weights = {'life_span': 0.5}
        elif game.level_number >= 20:
            fitness_weights = {'life_span': 0}
        else:
            fitness_weights = {}
        options={'push' : [game.level_number*100, game.level_number*-50], 'fitness_weights' : fitness_weights}
        print(options)
        game.calculate(options=options)
        if ((game.level_number >= 22) or (game.population.generation == 150)):
            for species_key in game.population.species_dict:
                game.population.species_dict[species_key][0].dump_connections(f'{species_key}.csv')
            break
        game.reproduce()

def replay(file_location):
    Game(pop_size=1).environment.replay(file_location)

class Game:
    def __init__(self, pop_size, logging_level='WARNING', log_run=False):
        logging.basicConfig(level=logging_level, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%H:%M:%S')
        self.population = Population(pop_size)
        self.environment = gym.make('LunarLanderContinuous-v2')
        self.input_nodes = self.environment.observation_space.shape[0] 
        try:
            self.output_nodes = self.environment.action_space.shape[0] # GAME DEPENDENT
            self.continuous = True
        except IndexError:
            self.output_nodes = 1
            self.continuous = False   
        self.population.create_population(self.input_nodes,self.output_nodes, init_connections= 8)#'all')
        self.log = pd.DataFrame() if log_run else None
        self.level_number = 0
        self.max_fitness = 0

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
        num_steps = 500
        for network in self.population:

            obs = self.environment.reset(seed=self.level_number, options=options)
            fitness = []
            for _ in range(num_steps):
                state = network.run_live(obs)
                action = self.actions(state)
                obs, rewards, done, info = self.environment.step(action)
                fitness.append(rewards)
                if done:
                    break
            network.fitness = max(sum(fitness),0.001)
            self.max_fitness = max(self.max_fitness, network.fitness)
            if network.fitness > 400:
                self.environment.log(f"replays/gen_{self.population.generation}_{str(network.id).split('-')[0]}.txt")
        if self.max_fitness > 400:
            self.max_fitness = 0
            self.level_number += 1
        if not (self.population.generation % 10): 
            self.environment.log(f'replays/sample_gen_{self.population.generation}.txt')
            
    def reproduce(self):
        if self.log is not None:
            self.log = self.log.append(self.population.log(self.level_number))
        self.population.advance_generation()
        logging.info(f'Generation: {self.population.generation}')
    
    def write_out(self, location):
        if self.log is not None:
            self.log.to_csv(location)

# %%
if __name__ == '__main__':
    main()




