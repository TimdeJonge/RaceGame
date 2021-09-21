#%%
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import os
# TODO :
# Speciation

#%% Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def filter_dict(nodes, filter_string):
    for key, val in nodes.items():
        if filter_string not in val:
            continue
        yield key

class AI_network():
    def __init__(self,  verbose=False):
        self.verbose = verbose
        self.fitness = 0
        self.results = []
        self.order = []
        self.species = None
    
    def create_network(self, n_input_nodes, n_output_nodes, innovation_df_g):
        total_nodes = n_input_nodes + n_output_nodes
        self.nodes = {}
        for i in range(total_nodes):
            node_type = 'Input' if i < n_input_nodes else 'Output'
            self.nodes[i] = node_type
        self.nodes[total_nodes] = 'Bias'
        self.connections  = pd.DataFrame(columns = ['From_node','To_node', 'Weight', 'Enabled', 'Innovation_number', 'Abbrev'])
        innovation_df_g = self.add_connection(innovation_df_g)
        self.order = list(filter_dict(self.nodes, 'Output'))
        return innovation_df_g

    def add_hidden_node(self, innovation_df_g, N_total_nodes):
        # Select random connection
        if any(self.connections['Enabled']):
            random_node_idx = self.connections.loc[(self.connections['Enabled'])].sample().index   
            self.connections.loc[random_node_idx,'Enabled'] = False
            # create hidden node
            self.nodes[N_total_nodes] = 'Hidden'
            if self.verbose:
                print(f'Hidden node added: {N_total_nodes}')
            # Connection to hidden
            innovation_df_g=self.append_connection(
                innovation_df_g, 
                From_node = self.connections.loc[random_node_idx, 'From_node'].item(),
                To_node = N_total_nodes,
                connection_weight = self.connections.loc[random_node_idx, 'Weight'].item()
            )

            # Connection hidden to
            innovation_df_g=self.append_connection(
                innovation_df_g,
                From_node = N_total_nodes,
                To_node = self.connections.loc[random_node_idx, 'To_node'].item(),
                connection_weight = 1
            )
            self.order.insert(0,N_total_nodes) 
            N_total_nodes += 1
        return innovation_df_g, N_total_nodes

    def add_connection(self, innovation_df_g):
        for _ in range(10):
            # From node can be any node
            From_node = random.choice(list(self.nodes))
            node_list = list(filter_dict(self.nodes, 'Output')) + list(filter_dict(self.nodes, 'Hidden'))
            To_node = random.choice(node_list)
            if self.connections.loc[(self.connections['From_node'] == From_node) & (self.connections['To_node'] == To_node)].empty:
                innovation_df_g = self.append_connection(innovation_df_g, From_node, To_node, connection_weight=np.random.randn())
                break         
        else:
            if self.verbose:
                print('Could not find a connection to add')
        return innovation_df_g
        
    def append_connection(self, innovation_df, From_node, To_node, connection_weight):
        if f'{From_node}-{To_node}' not in innovation_df['Abbrev'].values:
            innovation_df = innovation_df.append({'Abbrev' : f'{From_node}-{To_node}', 
                        'Innovation_number' : len(innovation_df)}, 
                        ignore_index=True)
        innovation_counter = innovation_df.loc[
                                innovation_df['Abbrev'] == f'{From_node}-{To_node}',
                                'Innovation_number'
                            ].values[0]
        self.connections = self.connections.append({'From_node' : From_node, 
                                        'To_node': To_node,  
                                        'Weight' : connection_weight,
                                        'Enabled': True, 
                                        'Innovation_number' : innovation_counter,
                                        'Abbrev' : f'{From_node}-{To_node}' 
                                        }, 
                                        ignore_index=True)
        if self.verbose:
            print(f'Connection added: from {From_node} to {To_node}')
        return innovation_df

    def mutate_weight_big(self, amount = 1):
        # Mutate random weight
        try:
            for _ in range(amount):
                random_node_idx = self.connections.loc[self.connections['Enabled']].sample().index[0]
                self.connections.loc[random_node_idx,'Weight']=np.random.randn()
                if self.verbose:
                    row = self.connections.loc[random_node_idx]
                    print(f"Weight adjusted from {row['From_node']} to {row['To_node']}, new weight: {self.connections.loc[random_node_idx,'Weight']}")
        except:
            if self.verbose:
                print('No connection enabled')

    def mutate_weight_small(self, amount = 1, factor = 0.1):
        # Small adjustment of existing weight
        try:
            for _ in range(amount):
                random_node_idx = self.connections.loc[self.connections['Enabled']].sample().index[0]
                self.connections.loc[random_node_idx,'Weight'] *= (1 + np.random.randn() * factor)
                if self.verbose:
                    row = self.connections.loc[random_node_idx]
                    print(f"Weight adjusted from {row['From_node']} to {row['To_node']}, new weight: {row['Weight']}")
        except:
            if self.verbose:
                print('No connection enabled')

    def disable_connection(self, amount = 1): 
        for _ in range(amount):
            try:
                random_node_idx = self.connections.loc[self.connections['Enabled']].sample().index[0]
                self.connections.loc[random_node_idx,'Enabled'] = False
                if self.verbose:
                    row = self.connections.loc[random_node_idx]
                    print(f"Disabled from {row['From_node']} to {row['To_node']}")
            except ValueError:
                if self.verbose:
                    print('No connection to disable')


    def enable_connection(self, amount = 1): # Assumes only 1 output node
        # Find disabled connections and enable
        for _ in range(amount):
            try:
                random_node_idx = self.connections.loc[self.connections['Enabled'] == False].sample().index[0]
                self.connections.loc[random_node_idx,'Enabled'] = True
                if self.verbose:
                    row = self.connections.loc[random_node_idx]
                    print(f"Enabled from {row['From_node']} to {row['To_node']}")
            except ValueError: 
                if self.verbose:
                    print('Nothing to enable')
            except KeyError as e:
                print(len(self.connections))
                print(f'connections={self.connections}')
                print(self.connections['Enabled'])
                print(~self.connections['Enabled'])
                raise(e)

    def build(self, total_nodes):
        self.state = np.ones(total_nodes+1)
        self.weights = np.zeros((total_nodes+1, total_nodes+1))
        for _, row in self.connections.loc[self.connections['Enabled']].iterrows():
            self.weights[row['From_node'], row['To_node']] = row['Weight']

    def run(self, df_input): 
        results = {}
        for counter, row in df_input.iterrows():
            self.state[:len(row)] = row
            for node in self.order:
                self.state[node] = sigmoid(np.dot(self.weights[:,node], self.state))
            results[counter] = self.state.copy()
        return pd.DataFrame.from_dict(results, orient='index')
    
    def run_live(self, row):
        self.state[:len(row)] = row
        for node in self.order:
            self.state[node] = sigmoid(np.dot(self.weights[:,node], self.state))

    def mutate(self, innovation_df_g, total_nodes):
        if random.random() < 0.8:
            for _ in range(len(self.nodes)):
                if random.random() < 0.9:
                    self.mutate_weight_small()
                if random.random() < 0.1:
                    self.mutate_weight_big()
        if random.random() < 0.1:
            innovation_df_g=self.add_connection(innovation_df_g)
        if random.random() < 0.01:
            innovation_df_g, total_nodes=self.add_hidden_node(innovation_df_g, total_nodes)
        if random.random() < 0.1:
            self.disable_connection()
        if random.random() < 0.05:
            self.enable_connection()

        self.build(total_nodes)
        return innovation_df_g, total_nodes

#%% 
class Population():
    def __init__(self, pop_size = 30):
        self.list = [AI_network() for _ in range(pop_size)]
        self.pop_size = pop_size
        self.generation = 0 
        self.champions = []
        self.innovation_df = pd.DataFrame(columns = ['Abbrev', 'Innovation_number'])
        self.species_count = 0
        self.total_nodes = None
    
    def __iter__(self):
        return iter(self.list)
    
    def __len__(self):
        return len(self.list)

    def create_population(self, n_input_nodes, n_output_nodes):
        self.total_nodes = n_input_nodes + n_output_nodes + 1
        for network in self.list:
            self.innovation_df = network.create_network(n_input_nodes, n_output_nodes, self.innovation_df)
            self.innovation_df = network.add_connection(self.innovation_df)
            self.innovation_df = network.add_connection(self.innovation_df)
            network.build(self.total_nodes)
        self.speciate()

    def speciate(self, distance_threshold=3):
        for network in self.list:
            for champion in self.champions:
                if self.distance(network, champion) < distance_threshold:
                    network.species = champion.species
                    break
            else:
                network.species = self.species_count
                self.species_count += 1
                self.champions.append(network)

    def advance_generation(self):
        fitness = {}
        species_dict = defaultdict(list)
        new_champions = []
        for champion in self.champions:
            for network in self.list:
                if network.species == champion.species:
                    species_dict[champion.species].append(network)
            if len(species_dict[champion.species]) > 0:
                fitness[champion.species] = np.mean([network.fitness for network in species_dict[champion.species]])
                species_dict[champion.species].sort(key = lambda x: (-x.fitness))
                new_champions.append(champion)

        self.champions = new_champions
        next_gen = []

        for species in fitness:
            species_dict[species] = species_dict[species][:4] #TODO: Make this more flexible
            if sum(fitness.values()) == 0:
                self.__init__()
                self.create_population(6,2) #TODO: Make this non-static
                return False
            new_size = fitness[species] / sum(fitness.values()) * self.pop_size
            if new_size > 3:
                next_gen.append(species_dict[species][0])
            for _ in range(int(new_size)):
                child = self.combine(*random.choices(species_dict[species], k=2))
                self.innovation_df, self.total_nodes = child.mutate(self.innovation_df, self.total_nodes)
                child.build(self.total_nodes)                
                next_gen.append(child)
        self.list = next_gen       
        self.speciate()
        self.generation += 1

    @staticmethod
    def distance(network1, network2, coeff_excess=1, coeff_disjoint=1, coeff_weight=0.4):
        size = max(len(network1.connections), len(network2.connections))
        if size < 20:
            size = 1
        excess_threshold = min(
                            network1.connections['Innovation_number'].max(),
                            network2.connections['Innovation_number'].max()
                            )

        total_connections = network1.connections.merge(
                                network2.connections, 
                                on='Innovation_number', 
                                how='outer', 
                                suffixes=[1,2])
        total_connections['Weight_diff'] = abs(total_connections['Weight1'] - total_connections['Weight2'])
        total_connections['Excess'] = total_connections['Innovation_number'] > excess_threshold
        total_connections['Disjoint'] = total_connections['Weight_diff'].isnull() & ~total_connections['Excess']
        if np.isnan(total_connections['Weight_diff'].mean()):
            weight_diff = 1
        else: 
            weight_diff = total_connections['Weight_diff'].mean()
        distance = (
                    coeff_weight * weight_diff + 
                    coeff_excess * total_connections['Excess'].sum() / size + 
                    coeff_disjoint * total_connections['Disjoint'].sum() / size 
                    )   
        return distance

    @staticmethod
    def residual_connections(connections, new_connections):
        not_in_new = connections.loc[~connections['Innovation_number'].isin(new_connections['Innovation_number'])]
        random_num = random.randint(0,len(not_in_new))
        if (random_num > 0 & len(not_in_new) > 0):
            add_connections = not_in_new.sample(random_num)
            new_connections = pd.concat((new_connections,add_connections), ignore_index=True)
        return new_connections


    @staticmethod
    def combine(network1, network2, verbose = False):
        connections1 = network1.connections
        connections2 = network2.connections
        matching_genes = connections1.loc[connections1['Innovation_number'].isin(connections2['Innovation_number'])]
        matching_genes = matching_genes.assign(rand_num=np.random.randint(1,3, matching_genes.shape[0]))
        match_in1 = connections1.loc[connections1['Innovation_number'].isin(matching_genes.loc[(matching_genes['rand_num'] == 1), 'Innovation_number'])]
        match_in2 = connections2.loc[connections2['Innovation_number'].isin(matching_genes.loc[(matching_genes['rand_num'] == 2), 'Innovation_number'])]
        new_connections = pd.concat((match_in1,match_in2))
        for _, row in new_connections.iterrows():
            if ((connections1.loc[(connections1['Innovation_number'] == row['Innovation_number']),'Enabled'].values + 
                connections2.loc[(connections2['Innovation_number'] == row['Innovation_number']),'Enabled'].values) == 1):
                new_connections.loc[new_connections['Innovation_number'] == row['Innovation_number'], 'Enabled'] = np.random.random() < 0.25
        new_network = AI_network(verbose = verbose)
        if (network1.fitness > network2.fitness):
            new_network.connections = population.residual_connections(connections1, new_connections)
        else:
            new_network.connections = population.residual_connections(connections2, new_connections)
        new_nodes = network1.nodes.copy()
        new_nodes.update(network2.nodes)
        new_network.nodes = {
                k : v 
                for k,v in new_nodes.items() 
                if (
                    (k in new_connections['To_node'].values) or 
                    (k in new_connections['From_node'].values) or 
                    (v != 'Hidden')
                )}
        new_network.new_order = list(np.unique(np.concatenate((network1.order, network2.order))))
        new_network.fitness = (network1.fitness + network2.fitness)/2
        return new_network
#%%
if __name__ == '__main__':
    population = Population()
    population.create_population(5,1)

    for network in population:
        for _ in range(10):
            population.innovation_df, population.total_nodes = network.mutate(population.innovation_df, population.total_nodes)
        network.run_live([1,1,1,1,1])
        network.fitness = network.state[5]
    for network in population:
        print(network.fitness)
    population.advance_generation()
    print(len(population.champions))
    print(population.champions[0].connections)
    print(population.champions[1].connections)
    print(population.distance(population.champions[0], population.champions[1]))

#%% Neural network 
if __name__ == '__main__':
    # symbol = 'BTCEUR'
    # markets = os.listdir('Data new/')
    # key = random.choice(markets)
    # df = pd.read_csv(f'Data new/{key}', index_col=0).fillna(0)
    # metrics = df.loc[:,'STD24':] #FIXME: Hard-coding this is scary
    # norm_metrics = ((metrics - metrics.mean()) / metrics.std())
    # df=df.loc[:, 'Timestamp':'symbol'].join(norm_metrics)
    # df = df.loc[500:,:]
    # batchSize = 100
    # trainingsdata = df.iloc[:batchSize, 5:9]
    innovation_df_g = pd.DataFrame(columns = ['Abbrev', 'Innovation_number'])
    N_inputs = 6
    N_output_nodes = 2
    network1 = AI_network(verbose = False)
    network2 = AI_network(verbose = False)
    N_total_nodes = N_inputs + N_output_nodes + 1
    innovation_df_g = network1.create_network(N_inputs, N_output_nodes, innovation_df_g) 
    innovation_df_g = network2.create_network(N_inputs, N_output_nodes, innovation_df_g) 
    innovation_df_g, N_total_nodes = network1.add_hidden_node(innovation_df_g, N_total_nodes)
    innovation_df_g, N_total_nodes = network2.add_hidden_node(innovation_df_g, N_total_nodes)
    innovation_df_g = network2.add_connection(innovation_df_g)
    innovation_df_g = network1.add_connection(innovation_df_g)
    new_network = Population.combine(network1, network2)
    network1.mutate_weight_small(amount = 10)
    network1.mutate_weight_big()
    network1.disable_connection(amount = 1)
    network1.enable_connection(amount = 1)
    network1.mutate(innovation_df_g, N_total_nodes)
    network1.build(N_total_nodes)


    # df_result=network1.run(trainingsdata)
    # gen = trainingsdata.iterrows()

    # print(network1.state)
    # _, row = next(gen)
    # print(row)
    # network1.run_live(row)
    # print(network1.state)
    # _, row = next(gen)
    # print(row)
    # network1.run_live(row)
    # print(network1.state)

    new_network= Population.combine(network1, network2)


# %%
