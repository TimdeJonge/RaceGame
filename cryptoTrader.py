#%%
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import os
from typing import Optional
from NeuralNet.neuralNet import Population


symbol = 'BTCEUR'
PLAYER_AMOUNT = 70
switch_fitness = 150 

markets = os.listdir('Data_bitcoin/')
key = random.choice(markets)
df = pd.read_csv(f'Data_bitcoin//{key}', index_col=0).fillna(0)
metrics = df.loc[:,'close_value':].drop(columns = 'symbol')
norm_metrics = ((metrics - metrics.mean()) / metrics.std())

#%%

class TradingBot:
    def __init__(self, metrics, verbose, logging):
        self.trading_fee = 0.0001
        self.data = metrics
        self.generation = 0
        self.population = Population(PLAYER_AMOUNT)
        self.input_nodes = len(metrics.columns)
        self.output_nodes = 1
        self.init_connections = 5
        self.output_node_number = self.input_nodes + self.output_nodes - 1
        self.population.create_population(self.input_nodes,self.output_nodes, self.init_connections)
        self.innovation_df = pd.DataFrame(columns = ['Abbrev', 'Innovation_number'])
        self.total_nodes = self.input_nodes + self.output_nodes + 1
        self.verbose = verbose
        self.max_fitness = 0
        self.logging = logging
        self.log = pd.DataFrame()

    def trade(self):
        while self.population.generation < 100:
            self.calculate()
            self.reproduce()
        
    def calculate(self):
        network_Score = []
        for _, network in enumerate(self.population):
            bank_account = 1000
            coins_wallet = 0
            trades = 0
            for _, row in metrics.loc[1000:,:].iterrows():
                network.run_live(row)
                for output_node in range(self.output_nodes):
                    output_node_index = self.input_nodes + output_node
                    if (
                        network.state[output_node_index] < 0.3
                        and bank_account > 15
                    ):
                        coins_wallet = bank_account / row['close_value'] * (1-self.trading_fee)
                        bank_account = 0
                        trades += 1
                    if (
                        network.state[output_node_index] > 0.7
                        and coins_wallet > 0.0001
                    ):
                        bank_account += coins_wallet * row['close_value'] * (1-self.trading_fee)
                        coins_wallet = 0
                        trades += 1
            
            change =  metrics.tail(1)['close_value'].values[0] / metrics.loc[1000,'close_value'] 
            network.fitness = int((bank_account + coins_wallet * row['close_value']) / change)
            network.trades = trades
            if self.verbose:
                print(f'network fitness: {network.fitness}')
            network_Score.append(network.fitness)
        self.max_fitness = max(network_Score)
        print(f'Maximal fitness of generation: {self.max_fitness}')
            
    def reproduce(self):
        if self.logging:
            self.log = self.log.append(self.population.log())
            self.log.to_csv('test1.csv')
        self.population.advance_generation(reduce_species=False)
        print(f'Generation: {self.population.generation}')


# %%
tradeBot = TradingBot(metrics, verbose = False, logging = False)
tradeBot.trade()

