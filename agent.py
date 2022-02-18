import pandas as pd

from probabilityEstimation import ConditionalProbability
from ocik.network import BayesianNetwork
from ocik import CausalLeaner
from drawing import draw, difference

import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\pakyr\\.conda\\envs\\bayesianEnv\\Library\\bin\\graphviz'


class Agent:
    def __init__(self, nodes, non_doable, edges, obs_data):
        self.nodes = nodes
        self.edges = edges
        self.non_doable = non_doable
        self.obs_data = obs_data
        self.conditional_prob = ConditionalProbability(self.obs_data, self.edges)
        self.bn = self.build_network()

    # If we are doing offline learning, when inserting a new node in the network we have to add also the new edges
    # in order to update the CPD table
    def update_values(self):
        self.conditional_prob = ConditionalProbability(self.obs_data, self.edges)
        self.bn = self.build_network()

    def build_network(self):
        # Define network structure
        bn = BayesianNetwork(self.edges)

        # Fill with conditional probabilities
        for node in bn.nodes():
            parents = []
            for edge in bn.edges():
                if node == edge[1]:
                    parents.append(edge[0])

            arr = self.conditional_prob.get_node_prob(node)
            # Invert the array for construction reasons
            arr = [arr[1], arr[0]]
            bn.set_cpd(node, arr, parents)
        return bn

    # Get non-duplicate nodes list from edges
    def nodes_from_edges(self, edges):
        nodes = []
        for edge in edges:
            nodes.append(edge[0])
            nodes.append(edge[1])
        return list(set(nodes))

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node):
        self.nodes.remove(node)

    def add_non_doable(self, node):
        self.non_doable.append(node)

    def remove_non_doable(self, node):
        self.non_doable.remove(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def remove_edge(self, edge):
        self.edges.remove(edge)

    def concatenate_data(self, data_to_concatenate, override=False):
        # Concatenate original data with received data
        # Pay attention on:
        # - identifier for each sample
        # - dimensions
        # - no duplicate data (as columns name)

        # Check if data are already present, if so decide to keep data already present or change with received new data
        drops = []
        for col in data_to_concatenate.columns:
            if col in self.obs_data.columns:
                drops.append(col)
        if override:
            # override old data
            self.obs_data.drop(columns=drops, inplace=True)
        else:
            # do not override old data
            data_to_concatenate.drop(columns=drops, inplace=True)

        # Merge of data based on the id
        self.obs_data = pd.merge(self.obs_data, data_to_concatenate, how='outer', on='id')

        # Decide how to manage NaN values

    def learning(self, parameters):
        # parameters = {
        #     'max_cond_vars': max_cond_vars,
        #     'do_size': do_size,
        #     'do_conf': do_conf,
        #     'ci_conf': ci_conf
        # }

        # Check for the updated values of CPD
        # self.update_values()
        # Learn
        estimator = CausalLeaner(self.nodes, non_dobale=self.non_doable, env=self.bn, obs_data=self.obs_data)
        model, undirected_edges = estimator.learn(max_cond_vars=parameters['max_cond_vars'], do_size=parameters['do_size'], do_conf=parameters['do_conf'], ci_conf=parameters['ci_conf'])

        return model, undirected_edges

    def print_structure(self):
        dot = draw(self.edges)
        dot.view(directory='tmp/2/')

    def build_request_msg(self, nodes_to_investigate: list, undirected_edges: list):
        # The message contains:
        #   - nodes with outliers values
        #   - nodes in undirected connections
        # In case of duplicates, eliminate them
        nodes_to_send = list(set(nodes_to_investigate + self.nodes_from_edges(undirected_edges)))

        # Build the list of non_doable
        non_doable = []
        for node in nodes_to_send:
            if node in self.non_doable:
                non_doable.append(node)

        # Data have to be passed when performing offline intervention simulations
        # Select only data referring to selected nodes and choose an amount of sample to send
        obs_data = self.obs_data
        data_to_send = obs_data.drop(columns=[x for x in obs_data.columns if x not in nodes_to_send])

        # Build message
        msg = dict()
        msg['nodes'] = list(nodes_to_send)
        msg['non_doable'] = list(non_doable)
        msg['data'] = data_to_send

        return msg

    def build_response_msg(self, discovered_edges: list):
        msg = dict()
        if discovered_edges:
            msg['edges'] = discovered_edges
            return msg
        else:
            return None

    def read_request(self, request_msg):
        # Extract nodes and data from request and add them to agent
        if request_msg:
            msg = request_msg
        else:
            return False  # Not going to learn

        # Check if all received nodes were already known: in this case it is useless to repeat the learning
        if all(item in self.nodes for item in msg['nodes']):
            print('Nodes already known, checking the previous learning results...')
            return False  # Not going to learn

        for node in msg['nodes']:
            if node not in self.nodes:
                self.add_node(node)

        for node in msg['non_doable']:
            if node not in self.non_doable:
                self.add_non_doable(node)

        #
        # self.concatenate_data(msg['data'])

        self.build_network()

        return True  # Going to learn

    def read_response(self, response):
        if response:
            # Read nodes and add to structure
            new_nodes = self.nodes_from_edges(response)
            for node in new_nodes:
                if node not in self.nodes:
                    self.add_node(node)
            # Read edges and add to structure
            for t in response:
                if t not in self.edges:
                    self.add_edge(t)

            return True

        return False







