#%%
import numpy as np
import pandas as pd
import random
import os
# TODO :
# From input node to final node
# any, all -> Random number of mutation 
# Reproduce
# Speciation
# Run
# Hidden node to hidden node -> disable all in between connections

#%% Functions
def sigmoid(x):
    #applying the sigmoid function
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
    
    def create_network(self, n_inputs_nodes, n_output_nodes, innovation_df_g):
        # How many nodes should be created
        count_input = n_inputs_nodes 
        count_output = count_input + n_output_nodes
        # Create dict of nodes
        node_dict = dict()
        total_nodes = count_output 
        for i in range(0,total_nodes):
            if i < count_input:
                node_type = 'Input'
            else:
                node_type = 'Output'
            node_dict[f'{i}'] =  node_type
        node_dict[str(total_nodes)] =  'Bias' # Bias node
        self.nodes = node_dict
        
        # Empty dataframe
        connections_df = pd.DataFrame(columns = ['From_node','To_node', 'Weight', 'Enabled', 'Innovation_number', 'Abbrev'])
        self.connections = connections_df

        # Create one random connection from input or bias to output

        innovation_df_g = self.add_connection(innovation_df_g)

        output_list = []
        for output_node in filter_dict(node_dict, 'Output'):
            output_list.append(int(output_node))
        self.order = output_list

        return innovation_df_g

    def add_hidden_node(self, innovation_df_g, N_total_nodes):
        nodes = self.nodes
        connections = self.connections
        order = self.order 

        # Select random connection
        try:
            rand_connection = connections.loc[(connections['Enabled'] == True)].sample().index[0]  
        except ValueError as e: 
            print(connections)        
            raise(e)                
        connections.loc[rand_connection,'Enabled'] = False

        # create hidden node
        new_node_number = N_total_nodes 
        node_type = 'Hidden'
        nodes[f'{new_node_number}'] =  node_type
        if self.verbose:
            print(f'Hidden node added: {new_node_number}')

        # Connection to hidden
        From_node = connections.loc[rand_connection, 'From_node']
        To_node = new_node_number
        connection_weight = connections.loc[rand_connection, 'Weight']

        innovation_df_g = innovation_df_g.append({'Abbrev' : f'{From_node}-{To_node}', 
                                            'Innovation_number' : len(innovation_df_g)}, 
                                                ignore_index=True)
        innovation_counter = innovation_df_g.loc[innovation_df_g['Abbrev'] == f'{From_node}-{To_node}',
                                                'Innovation_number'].values[0]
        connections =connections.append({'From_node' : int(From_node), 
                                        'To_node': int(To_node),  
                                        'Weight' : connection_weight,
                                        'Enabled': True, 
                                        'Innovation_number' : int(innovation_counter),
                                        'Abbrev' : f'{From_node}-{To_node}'
                                        }, 
                                        ignore_index=True)
        
        if self.verbose:
            print(f'Connection from: {From_node} to {To_node}')

        # Connection hidden to
        From_node = new_node_number
        To_node = connections.loc[rand_connection, 'To_node']
        connection_weight = 1
        innovation_df_g = innovation_df_g.append({'Abbrev' : f'{From_node}-{To_node}', 
                                            'Innovation_number' : len(innovation_df_g)}, 
                                                ignore_index=True)
        innovation_counter = innovation_df_g.loc[innovation_df_g['Abbrev'] == f'{From_node}-{To_node}',
                                                'Innovation_number'].values[0]
        connections =connections.append({'From_node' : int(From_node), 
                                        'To_node': int(To_node),  
                                        'Weight' : connection_weight,
                                        'Enabled': True, 
                                        'Innovation_number' : int(innovation_counter),
                                        'Abbrev' : f'{From_node}-{To_node}'
                                        }, 
                                        ignore_index=True)    

        if self.verbose:
             print(f'Connection from: {From_node} to {To_node}')

        self.order.insert(0,int(new_node_number))
        self.nodes = nodes
        self.connections = connections 
        N_total_nodes += 1
        return innovation_df_g, N_total_nodes

    def add_connection(self, innovation_df_g):
        nodes = self.nodes
        connections = self.connections

        loop_counter = 0
        new_connection = False
        while new_connection == False:
            # From node can be any node
            random_node = random.sample(nodes.keys(),1)[0]
            From_node = int(random_node)

            # To node can only be hidden and output
            node_list = []
            for hidden_nodes in filter_dict(nodes, 'Hidden'):
                node_list.append(hidden_nodes)

            for output_nodes in filter_dict(nodes, 'Output'):
                node_list.append(output_nodes)

            random_node = random.sample(node_list,1)[0]
            To_node = int(random_node)

            if (connections.loc[(connections['From_node'] == From_node) & (connections['To_node'] == To_node)].empty) == True:
                connection_weight = np.random.randn()
                if  innovation_df_g['Abbrev'].str.contains(f'{From_node}-{To_node}').any():
                    try: # The above if statement will also be true if '8-0' == '28-0'
                        innovation_df_g.loc[innovation_df_g['Abbrev'] == f'{From_node}-{To_node}',
                                                        'Innovation_number'].values[0]
                    except IndexError:
                        innovation_df_g = innovation_df_g.append({'Abbrev' : f'{From_node}-{To_node}', 
                                'Innovation_number' : len(innovation_df_g)}, 
                                    ignore_index=True) 
                else:
                    innovation_df_g = innovation_df_g.append({'Abbrev' : f'{From_node}-{To_node}', 
                            'Innovation_number' : len(innovation_df_g)}, 
                                ignore_index=True)
                innovation_counter = innovation_df_g.loc[innovation_df_g['Abbrev'] == f'{From_node}-{To_node}',
                                                        'Innovation_number'].values[0]
                connections =connections.append({'From_node' : From_node, 
                                                'To_node': To_node,  
                                                'Weight' : connection_weight,
                                                'Enabled': True, 
                                                'Innovation_number' : innovation_counter,
                                                'Abbrev' : f'{From_node}-{To_node}' # self.innovation_counter
                                                }, 
                                                ignore_index=True)
                new_connection == True
                if self.verbose:
                    print(f'Connection added: from {From_node} to {To_node}')
                break
            else: # Connection already exists
                loop_counter += 1
                if loop_counter == 10:
                    if self.verbose:
                        print('Could not find a connection to add')
                    break

        self.connections = connections 
        return innovation_df_g

    def mutate_weight_big(self, amount = 1):
        connections = self.connections

        # Mutate random weight
        try:
            for i in range(amount):
                random_node_idx = connections.loc[connections['Enabled'] == True].sample().index[0]
                new_connection_weight =  np.random.randn()
                connections.loc[(connections.index == random_node_idx),'Weight']=new_connection_weight
                if self.verbose:
                    from_n = connections.loc[random_node_idx,'From_node']
                    to_n = connections.loc[random_node_idx,'To_node']
                    print(f'Weight adjusted from {from_n} to {to_n}, new weight: {new_connection_weight}')
        except:
            if self.verbose:
                print('No connection enabled')
            pass
        self.connections = connections

    def mutate_weight_small(self, amount = 1, factor = 0.1):
        connections = self.connections

        # Small adjustment of existing weight
        try:
            for i in range(amount):
                random_node_idx = connections.loc[connections['Enabled'] == True].sample().index[0]
                old_connection_weight = connections.loc[(connections.index == random_node_idx),'Weight']
                new_connection_weight =  np.random.randn() * factor * old_connection_weight + old_connection_weight
                connections.loc[(connections.index == random_node_idx),'Weight']=new_connection_weight
                if self.verbose:
                    from_n = connections.loc[random_node_idx,'From_node']
                    to_n = connections.loc[random_node_idx,'To_node']
                    print(f'Weight adjusted from {from_n} to {to_n}, new weight: {new_connection_weight}')
        except:
            if self.verbose:
                print('No connection enabled')
            pass
        self.connections = connections

    def disable_connection(self, amount = 1): 
        nodes = self.nodes
        connections = self.connections

        for i in range(amount):
            try:
                random_connection = connections.loc[connections['Enabled'] == True].sample().index[0]
                connections.loc[random_connection,'Enabled'] = False
                if self.verbose:
                    from_n = connections.loc[random_connection,'From_node']
                    to_n = connections.loc[random_connection,'To_node']
                    print(f'Disabled from {from_n} to {to_n}')
            except ValueError:
                if self.verbose:
                    print('No connection to disable')
                pass

        self.connections = connections 

    def enable_connection(self, amount = 1): # Assumes only 1 output node
        nodes = self.nodes
        connections = self.connections

        # Find disabled connections and enable
        for i in range(amount):
            try:
                random_connection = connections.loc[connections['Enabled'] == False].sample().index[0]
                connections.loc[random_connection,'Enabled'] = True
                if self.verbose:
                    from_n = connections.loc[random_connection,'From_node']
                    to_n = connections.loc[random_connection,'To_node']
                    print(f'Enabled from {from_n} to {to_n}')
            except ValueError: 
                if self.verbose:
                    print('No to enable')
                pass

        self.connections = connections 

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

    def procreate(self, innovation_df_g, total_nodes):
        for _ in range(5):
            if random.random() < 0.9:
                self.mutate_weight_small()
        if random.random() < 0.4:
            innovation_df_g=self.add_connection(innovation_df_g)
        if random.random() < 0.1:
            innovation_df_g, total_nodes=self.add_hidden_node(innovation_df_g, total_nodes)
        if random.random() < 0.1:
            self.disable_connection()
        if random.random() < 0.2:
            self.enable_connection()
        if random.random() < 0.3:
            self.mutate_weight_big()
        self.build(total_nodes)
        return innovation_df_g, total_nodes

#%% 
class reproduction:
    def combine(network1, network2, verbose = False):
        connections1 = network1.connections
        connections2 = network2.connections

        # Only use enabled connections
        # connections1 = connections1.loc[connections1['Enabled'] == True]
        # connections2 = connections2.loc[connections2['Enabled'] == True]

        # Connections that both networks have in common
        double_connections = connections1.loc[connections1['Innovation_number'].isin(connections2['Innovation_number'])]
        double_connections.loc[:,'rand_num'] = np.random.randint(1,3, double_connections.shape[0])

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
        random_num = random.randint(0,len(connections2))
        try:
            add_connections = connections2_notin.sample(random_num)
        except ValueError:
            if verbose:
                print('No value to add')
        new_connections = pd.concat((new_connections,add_connections), ignore_index=True)

        # Set new nodes
        new_nodes = network1.nodes
        new_nodes.update(network2.nodes)

        new_order = np.unique(np.concatenate((network1.order, network2.order)))

        new_network = AI_network(verbose = verbose)
        new_network.connections = new_connections
        new_network.nodes = new_nodes
        new_network.order = new_order

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

    N_inputs = 5
    N_output_nodes = 1

    innovation_counter = 0

    network1 = AI_network(verbose = True)
    network2 = AI_network(verbose = True)

    N_total_nodes = N_inputs + N_output_nodes + 1
    innovation_df_g = network1.create_network(N_inputs, N_output_nodes, innovation_df_g) # DF to keep track of all innovation numbers
    # print(innovation_df_g)
    # print(nodes_dict_g)
    # print(network1.nodes)
    # print(network1.connections)
    # print(network1.order)

    innovation_df_g = network2.create_network(N_inputs, N_output_nodes, innovation_df_g) # DF to keep track of all innovation numbers
    # print(innovation_df_g)
    # print(nodes_dict_g)
    # print(network2.nodes)
    # print(network2.connections)
    # print(network2.order)

    innovation_df_g, N_total_nodes = network1.add_hidden_node(innovation_df_g, N_total_nodes)
    print(innovation_df_g)
    print(network1.nodes)
    print(network1.connections)
    print(network1.order)

    innovation_df_g, N_total_nodes = network2.add_hidden_node(innovation_df_g, N_total_nodes)
    print(innovation_df_g)

    print(network2.connections)
    print(network2.order)
    print(network2.nodes)


    innovation_df_g = network2.add_connection(innovation_df_g)
    print(innovation_df_g)
    print(network2.nodes)
    print(network2.connections)

    innovation_df_g,N_total_nodes = network1.add_hidden_node(innovation_df_g, N_total_nodes)
    print(innovation_df_g)
    print(network1.nodes)
    print(network1.connections)
    print(network1.order)

    innovation_df_g = network1.add_connection(innovation_df_g)
    print(innovation_df_g)
    print(network1.nodes)
    print(network1.connections)


    new_network = reproduction.combine(network1, network2)
    print(new_network.nodes)
    print(new_network.connections)
    print(new_network.order)


    network1.mutate_weight_small(amount = 10)
    print(network1.connections)

    network1.mutate_weight_big()
    print(network1.connections)


    network1.disable_connection(amount = 1)
    print(network1.connections)

    network1.enable_connection(amount = 1)
    print(network1.connections)


    network1.build(N_total_nodes)
    print(network1.state)
    print(network1.weights)

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

    print(new_network.connections)
    print(new_network.nodes)
    print(new_network.order)


# %%

# fig = go.Figure(data=[go.Candlestick(x=df['open_date'],
#             open=df['open_value'],
#             high=df['high_value'],
#             low=df['low_value'],
#             close=df['close_value'])])
# fig.show()



# import numpy as np
# from scipy import integrate
# import plotly.express as px

# close=df['close_value']
# intClose = np.zeros(len(close))
# for i in range(1, len(close)):
#     j = i - 1
#     intClose[i] = close[j] - close[i]


# fig = px.line(x=df['open_date'], y=intClose)
# fig.show()


#%%



# %%
# Create dataframe of connections
# for counter_output in range(n_output_nodes): # Input to hidden nodes
#     To_node = counter_output + count_input

#     for counter_input in range(n_inputs_nodes):
#         connection_weight = np.random.randn()
#         From_node = counter_input 
#         if  innovation_df_g['Abbrev'].str.contains(f'{From_node}-{To_node}').any():
#             pass
#         else:
#             innovation_df_g = innovation_df_g.append({'Abbrev' : f'{From_node}-{To_node}', 
#                                                 'Innovation_number' : len(innovation_df_g)}, 
#                                                     ignore_index=True)

#         innovation_counter = innovation_df_g.loc[innovation_df_g['Abbrev'] == f'{From_node}-{To_node}',
#                                                 'Innovation_number'].values[0]
#         connections_df =connections_df.append({'From_node' : int(From_node), 
#                                                 'To_node': int(To_node),  
#                                                 'Weight' : connection_weight,
#                                                 'Enabled': True, 
#                                                 'Innovation_number' : int(innovation_counter),
#                                                 'Abbrev' : f'{From_node}-{To_node}'
#                                                 }, 
#                                                 ignore_index=True)
            
#     # Connect bias to output nodes
#     connection_weight = np.random.randn() # Bias weight
#     From_node = total_nodes
#     if  innovation_df_g['Abbrev'].str.contains(f'{From_node}-{To_node}').any():
#             pass
#     else:
#         innovation_df_g = innovation_df_g.append({'Abbrev' : f'{From_node}-{To_node}', 
#                                             'Innovation_number' : len(innovation_df_g)}, 
#                                                 ignore_index=True)
#     innovation_counter = innovation_df_g.loc[innovation_df_g['Abbrev'] == f'{From_node}-{To_node}',
#                                             'Innovation_number'].values[0]
#     connections_df =connections_df.append({'From_node' : int(From_node), 
#                                             'To_node': int(To_node),  
#                                             'Weight' : connection_weight,
#                                             'Enabled': True, 
#                                             'Innovation_number' : int(innovation_counter),
#                                             'Abbrev' : f'{From_node}-{To_node}'
#                                             }, 
#                                             ignore_index=True)