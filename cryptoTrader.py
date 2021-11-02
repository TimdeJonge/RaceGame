# TODO: testing function -> data that hasn't been used before
# TODO: Visualise trades

#%%
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import os
from typing import Optional
from NeuralNet.neuralNet import Population


symbol = 'BTCEUR'
PLAYER_AMOUNT = 50

class TradingBot:
    def __init__(self, verbose, logging):
        self.trading_fee = 0.001
        data = self.select_market()
        self.generation = 0
        self.population = Population(PLAYER_AMOUNT)
        self.input_nodes = len(data.columns)
        self.output_nodes = 1
        self.init_connections = 15
        self.output_node_number = self.input_nodes + self.output_nodes - 1
        self.population.create_population(self.input_nodes,self.output_nodes, self.init_connections)
        self.innovation_df = pd.DataFrame(columns = ['Abbrev', 'Innovation_number'])
        self.total_nodes = self.input_nodes + self.output_nodes + 1
        self.verbose = verbose
        self.max_fitness = 0
        self.logging = logging
        self.log = pd.DataFrame()
        self.profit = 50

    def select_market(self):
        markets = os.listdir('Data_bitcoin/Bull_market/')
        key = random.choice(markets)
        df = pd.read_csv(f'Data_bitcoin/Bull_market//{key}', index_col=0).fillna(0)
        metrics = df.loc[1000:,'close_value':].drop(columns = 'symbol')
        metrics.loc[:,'STD24':] = ((metrics.loc[:,'STD24':] - metrics.loc[:,'STD24':].mean()) / metrics.loc[:,'STD24':].std())
        size_dataset = 3000
        first_sample = np.random.randint(0,len(metrics)-size_dataset)
        return  metrics[first_sample:first_sample+size_dataset].reset_index()

    def trade(self):
        while self.population.generation < 100:
            network_Score = self.calculate()
            self.reproduce()
            if np.mean(network_Score[:4] ) > self.profit: 
                self.select_market()
        self.evaluate()

    def select_data(self):
        dataset1 = self.select_market()
        dataset2 = self.select_market()
        dataset3 = self.select_market()
        dataset4 = self.select_market()
        dataset5 = self.select_market()
        return  [dataset1,dataset2,dataset3, dataset4,dataset5]

    def calculate(self):
        network_Score = []
        datasets = self.select_data()
        for _, network in enumerate(self.population):
            network_fitness = []
            for dataset in datasets:
                network = self.compute(network, dataset)
                if self.verbose:
                    print(f'network fitness: {network.fitness}')
                network_fitness.append(network.fitness)
            network.fitness = max(0.01,np.mean(network_fitness))
            network_Score.append(network.fitness)
        self.max_fitness = max(network_Score)
        print(f'Maximal fitness of generation: {self.max_fitness}')
        return network_Score

    def evaluate(self):
        for _ in range(5):
            datasets = self.select_data()
            for num, network in enumerate(self.population.champions):
                network_fitness = []
                network_Score = []
                for dataset in datasets:
                    network = self.compute(network, dataset, evaluate = True)
                    if self.verbose:
                        print(f'network fitness: {network.fitness}')
                    network_fitness.append(network.fitness)
                network_Score.append(np.mean(network_fitness))
                print(f'Money made by champion {num}: {network_Score[0]}')

    def compute(self, network, dataset, evaluate = None):
        bank_account = 1000
        coins_wallet = 0
        trades = 0
        for _, row in dataset.iterrows():
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
        change =  np.mean(dataset.tail(100)['close_value'].values) / np.mean(dataset.loc[0:100,'close_value'].values )
        network.fitness = int((bank_account + coins_wallet * row['close_value']) / change)
        network.trades = trades
        return network

    def reproduce(self):
        if self.logging:
            self.log = self.log.append(self.population.log())
            self.log.to_csv('test1.csv')
        self.population.advance_generation(reduce_species=False)
        print(f'Generation: {self.population.generation}')


# %%
tradeBot = TradingBot(verbose = False, logging = False)
tradeBot.trade()



# %%
