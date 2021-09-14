#%%
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
    
    def create_network(self, n_input_nodes, n_output_nodes, innovation_df_g):
        total_nodes = n_input_nodes + n_output_nodes
        self.nodes = dict()
        for i in range(total_nodes):
            if i < n_input_nodes:
                node_type = 'Input'
            else:
                node_type = 'Output'
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
            self.append_connection(
                innovation_df_g, 
                From_node = self.connections.loc[random_node_idx, 'From_node'].item(),
                To_node = N_total_nodes,
                connection_weight = self.connections.loc[random_node_idx, 'Weight'].item()
            )

            # Connection hidden to
            self.append_connection(
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
        if not f'{From_node}-{To_node}' in innovation_df['Abbrev'].values:
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
        if random.random() < 0.05:
            innovation_df_g=self.add_connection(innovation_df_g)
        if random.random() < 0.01:
            innovation_df_g, total_nodes=self.add_hidden_node(innovation_df_g, total_nodes)
        if random.random() < 0.06:
            self.disable_connection()
        if random.random() < 0.03:
            self.enable_connection()

        self.build(total_nodes)
        return innovation_df_g, total_nodes

#%% 
class reproduction:
    def combine(network1, network2, verbose = False):
        connections1 = network1.connections
        connections2 = network2.connections

        # Connections that both networks have in common
        double_connections = connections1.loc[connections1['Innovation_number'].isin(connections2['Innovation_number'])]
        double_connections.loc[:,'rand_num'] = np.random.randint(1,3, double_connections.shape[0])
        # double_connections.loc[:,'rand_num'] = double_connections.apply()

        innov1 = double_connections.loc[(double_connections['rand_num'] == 1), 'Innovation_number']
        double_con1 = connections1.loc[connections1['Innovation_number'].isin(innov1)]

        innov2 = double_connections.loc[(double_connections['rand_num'] == 2), 'Innovation_number']
        double_con2 = connections2.loc[connections2['Innovation_number'].isin(innov2)]

        new_connections = pd.concat((double_con1,double_con2))

        # Find connections that are not yet in new connections
        connections1_notin = connections1.loc[~connections1['Innovation_number'].isin(new_connections['Innovation_number'])]

        # Randomly add connections that were not in both networks
        random_num = random.randint(0,len(connections1_notin))
        try:
            add_connections = connections1_notin.sample(random_num)
        except ValueError:
            if verbose:
                print('No value to add')
        new_connections = pd.concat((new_connections,add_connections), ignore_index=True)
        # Same story
        connections2_notin = connections2.loc[~connections2['Innovation_number'].isin(new_connections['Innovation_number'])]
        random_num = random.randint(0,len(connections2_notin))
        try:
            add_connections = connections2_notin.sample(random_num)
        except ValueError:
            if verbose:
                print('No value to add')
        new_connections = pd.concat((new_connections,add_connections), ignore_index=True)
        # Set new nodes
        new_nodes = network1.nodes
        new_nodes.update(network2.nodes)

        new_order = list(np.unique(np.concatenate((network1.order, network2.order))))

        new_network = AI_network(verbose = verbose)
        new_network.connections = new_connections
        new_network.nodes = new_nodes
        new_network.order = new_order
        new_network.fitness = (network1.fitness + network2.fitness)/2
        return new_network

    def speciate(population):
        # Some code to differentiate between networks
        # Some code to assign a species number to members
        return population

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
    innovation_df_g,N_total_nodes = network1.add_hidden_node(innovation_df_g, N_total_nodes)
    innovation_df_g = network1.add_connection(innovation_df_g)
    new_network = reproduction.combine(network1, network2)
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

    new_network= reproduction.combine(network1, network2)


# %%
