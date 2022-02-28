#%%
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import logging
import uuid

#%% Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def filter_dict(nodes, filter_string):
    for key, val in nodes.items():
        if filter_string not in val:
            continue
        yield key

class AI_network():
    def __init__(self, population):
        self.fitness = 0
        self.results = []
        self.order = []
        self.species = None
        self.population = population
        self.id = str(uuid.uuid4()).split('-')[0]
        self.parents = (None, None)

    def create_network(self, n_input_nodes, n_output_nodes, init_connections = 1):
        total_nodes = n_input_nodes + n_output_nodes
        self.nodes = {}
        for i in range(total_nodes):
            node_type = 'Input' if i < n_input_nodes else 'Output'
            self.nodes[i] = node_type
        self.nodes[total_nodes] = 'Bias'
        self.connections  = pd.DataFrame(columns = ['From_node','To_node', 'Weight', 'Enabled', 'Innovation_number', 'Abbrev'])
        if init_connections == 'all':
            for input in range(n_input_nodes):
                for output in range(n_input_nodes, n_input_nodes + n_output_nodes):
                    self.append_connection(input, output, connection_weight=np.random.randn())
            for output in range(n_input_nodes, n_input_nodes + n_output_nodes):
                self.append_connection(total_nodes, output, connection_weight=np.random.randn())
        else:
            for _ in range(init_connections):
                self.add_connection()
        self.order = list(filter_dict(self.nodes, 'Output'))


    def add_hidden_node(self):
        # Select random connection
        if any(self.connections['Enabled']):
            random_node_idx = self.connections.loc[(self.connections['Enabled'])].sample().index   
            self.connections.loc[random_node_idx,'Enabled'] = False

            # create hidden node
            new_node = self.population.add_node()
            self.nodes[new_node] = 'Hidden'
            logging.debug(f'Hidden node added: {new_node}')

            # Connection to hidden
            self.append_connection(
                From_node = self.connections.loc[random_node_idx, 'From_node'].item(),
                To_node = new_node,
                connection_weight = self.connections.loc[random_node_idx, 'Weight'].item()
            )

            # Connection hidden to
            self.append_connection(
                From_node = new_node,
                To_node = self.connections.loc[random_node_idx, 'To_node'].item(),
                connection_weight = 1
            )

            self.order.insert(0,new_node) 

    def add_connection(self):
        for _ in range(10):
            # From node can be any node
            From_node = random.choice(list(self.nodes))
            node_list = list(filter_dict(self.nodes, 'Output')) + list(filter_dict(self.nodes, 'Hidden'))
            To_node = random.choice(node_list)
            if self.connections.loc[(self.connections['From_node'] == From_node) & (self.connections['To_node'] == To_node)].empty:
                self.append_connection(From_node, To_node, connection_weight=np.random.randn())
                break         
        else:
            logging.debug('Could not find a connection to add')
        
    def append_connection(self, From_node, To_node, connection_weight):
        connection = {  
                        'From_node' : From_node, 
                        'To_node': To_node,  
                        'Weight' : connection_weight,
                        'Enabled': True,
                        'Abbrev' : f'{From_node}-{To_node}' 
                    }
        innovation_number = self.population.check_innovation(connection)
        connection.update(Innovation_number=innovation_number)
        self.connections = self.connections.append(connection, ignore_index=True)
        logging.debug(f'Connection added: from {From_node} to {To_node}')

    def mutate_weight_big(self, amount = 1):
        # Mutate random weight
        try:
            for _ in range(amount):
                random_node_idx = self.connections.loc[self.connections['Enabled']].sample().index[0]
                self.connections.loc[random_node_idx,'Weight']=np.random.randn()
                row = self.connections.loc[random_node_idx]
                logging.debug(f"Weight adjusted from {row['From_node']} to {row['To_node']}, new weight: {self.connections.loc[random_node_idx,'Weight']}")
        except:
            logging.debug('No connection enabled')

    def mutate_weight_small(self, amount = 1, factor = 0.1):
        # Small adjustment of existing weight
        try:
            for _ in range(amount):
                random_node_idx = self.connections.loc[self.connections['Enabled']].sample().index[0]
                self.connections.loc[random_node_idx,'Weight'] *= (1 + np.random.randn() * factor)
                row = self.connections.loc[random_node_idx]
                logging.debug(f"Weight adjusted from {row['From_node']} to {row['To_node']}, new weight: {row['Weight']}")
        except:
                logging.debug('No connection enabled')

    def disable_connection(self, amount = 1): 
        for _ in range(amount):
            try:
                random_node_idx = self.connections.loc[self.connections['Enabled']].sample().index[0]
                self.connections.loc[random_node_idx,'Enabled'] = False
                row = self.connections.loc[random_node_idx]
                logging.debug(f"Disabled from {row['From_node']} to {row['To_node']}")
            except ValueError:
                logging.debug('No connection to disable')

    def enable_connection(self, amount = 1): # Assumes only 1 output node
        # Find disabled connections and enable
        for _ in range(amount):
            try:
                random_node_idx = self.connections.loc[self.connections['Enabled'] == False].sample().index[0]
                self.connections.loc[random_node_idx,'Enabled'] = True
                row = self.connections.loc[random_node_idx]
                logging.debug(f"Enabled from {row['From_node']} to {row['To_node']}")
            except ValueError: 
                logging.debug('Nothing to enable')
            except KeyError as e:
                logging.error(len(self.connections))
                logging.error(f'connections={self.connections}')
                logging.error(self.connections['Enabled'])
                logging.error(~self.connections['Enabled'])
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
        return self.state

    def mutate(self):
        if random.random() < 0.8:
            for _ in range(len(self.nodes)):
                if random.random() < 0.9:
                    self.mutate_weight_small()
                if random.random() < 0.1:
                    self.mutate_weight_big()
        if random.random() < 0.05:
            self.add_connection()
        if random.random() < 0.01:
            self.add_hidden_node()
        if random.random() < 0:
            self.disable_connection()
        if random.random() < 0:
            self.enable_connection()

        self.build(self.population.total_nodes)
    
    def log(self):
        return {
            'id' : self.id,
            'species' : self.species,
            'fitness' : self.fitness,
            'parent1' : self.parents[0],
            'parent2' : self.parents[1],
            'connections' : len(self.connections),
            'enabled_connections' : len(self.connections.loc[self.connections['Enabled']]),
            'disabled_connections' : len(self.connections.loc[self.connections['Enabled'] == False]),
            'hidden_nodes' : len(list(filter_dict(self.nodes, 'Hidden')))
        }
    
    def dump_connections(self, path):
        self.connections.to_csv(path)

#%% 
class Population():
    def __init__(self, pop_size = 30):
        self.list = [AI_network(self) for _ in range(pop_size)]
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

    def check_innovation(self, connection):
        if f"{connection['Abbrev']}" not in self.innovation_df['Abbrev'].values:
            connection.update(Innovation_number=len(self.innovation_df.index))
            self.innovation_df = self.innovation_df.append(connection, ignore_index=True)
        return self.innovation_df.loc[
                                self.innovation_df['Abbrev'] == connection['Abbrev'],
                                'Innovation_number'
                            ].values[0]
    
    def add_node(self):
        self.total_nodes += 1
        return self.total_nodes
        
    def create_population(self, n_input_nodes, n_output_nodes, init_connections):
        self.total_nodes = n_input_nodes + n_output_nodes + 1
        for network in self.list:
            network.create_network(n_input_nodes, n_output_nodes, init_connections)
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

    def advance_generation(self, reduce_species=False):
        fitness = {}
        species_dict = defaultdict(list)
        new_champions = []
        for champion in self.champions:
            species_key = champion.species
            for network in self.list:
                if network.species == species_key:
                    species_dict[species_key].append(network)
            if len(species_dict[species_key]) > 0:
                fitness[species_key] = np.mean([network.fitness for network in species_dict[species_key]])
                if fitness[species_key] < 0:
                    raise ValueError('Fitness needs to be positive for this algorithm to work.')
                species_dict[species_key].sort(key = lambda x: (-x.fitness))
                new_champions.append(champion)
        self.species_dict = species_dict.copy()
        self.champions = new_champions
        next_gen = []
        # if reduce_species:
        #     print('Reducing species!')
        #     print('Old species:')
        #     for species in fitness:
        #         print(species, fitness[species])
        #     fitness = {k : v for k, v in fitness.items() if k in sorted(fitness, key=fitness.get, reverse=True)[:3]}
        #     print('New species:')
        #     for species in fitness:
        #         print(species, fitness[species])
        fitwork = max(self.list, key=lambda x: x.fitness)
        next_gen.append(fitwork)
        species_log_dict = {}
        for species_key in fitness:
            species = species_dict[species_key]
            new_size = fitness[species_key] / sum(fitness.values()) * self.pop_size
            champ_log = species[0].log()
            species_object = {
                'species' : species_key,
                'members' : len(species),
                'new_size' : new_size,
                'avg_fitness' : fitness[species_key],
                'top_performer' : champ_log['id'],
                'top_fitness' : champ_log['fitness'],
            }
            species_log_dict[species_key] = species_object
            if new_size >= 3:
                next_gen.append(species[0])
            for _ in range(int(new_size)):
                temperature = self.generation / 100
                parent1, parent2 = random.choices(
                    species, 
                    k=2, 
                    weights=[np.exp(network.fitness*temperature) for network in species]
                )
                child = self.combine(parent1, parent2)
                next_gen.append(child)

        for network in next_gen:
            network.build(self.total_nodes)
        self.list = next_gen
        self.speciate(distance_threshold=2.5)
        self.generation += 1
        return species_log_dict

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

    def combine(self, network1, network2, mutate=True):
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
        child = AI_network(self)
        if (network1.fitness > network2.fitness):
            child.connections = pd.concat((new_connections,connections1.loc[~connections1['Innovation_number'].isin(new_connections['Innovation_number'])]), ignore_index=True)
        else:
            child.connections = pd.concat((new_connections,connections2.loc[~connections2['Innovation_number'].isin(new_connections['Innovation_number'])]), ignore_index=True)
        child_nodes = network1.nodes.copy()
        child_nodes.update(network2.nodes)
        child.nodes = {
                k : v 
                for k,v in child_nodes.items() 
                if (
                    (k in new_connections['To_node'].values) or 
                    (k in new_connections['From_node'].values) or 
                    (v != 'Hidden')
                )}
        child.order = list(np.unique(np.concatenate((network1.order, network2.order))))
        child.fitness = (network1.fitness + network2.fitness)/2
        child.parents = (network1.id, network2.id)
        if mutate:
            child.mutate()      
        return child
    
    def log(self):
        # pd.DataFrame.from_dict({id(network) : network.log() for network in self}, orient = 'index').assign(generation = self.generation)
        logfile = pd.DataFrame([network.log() for network in self]).assign(generation = self.generation)
        return logfile.set_index(['generation', 'id'])
    
    def test_logging(self, level):
        logging.log(level, 'PANIC')
        
         
#%%
#Case to test: Make 2 networks, combine them, see what innovation_df looks like throughout.
if __name__ == '__main__':
    population = Population()
    population.create_population(5,1, 5)

    for network in population:
        for _ in range(10):
            network.mutate()
        network.run_live([1,1,1,1,1])
        network.fitness = network.state[5]

    population.advance_generation()
    print(len(population.champions))
    print(population.champions[0].connections)


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
