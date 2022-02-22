import pandas as pd
from utils.probabilityEstimation import ConditionalProbability
from ocik.network import BayesianNetwork
from ocik import CausalLeaner
from utils.drawing import draw

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
        self.undirected_edges = []
        self.incomplete = []

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

    def add_undirected_edges(self, undirected_edges):
        for edge in undirected_edges:
            self.undirected_edges.append(edge)

    def concatenate_data(self, data_to_concatenate, override=True):
        # Concatenate original data with received data
        # Pay attention on:
        # - identifier for each sample
        # - dimensions
        # - no duplicate data (as columns name)

        # When a column is already present, decide if keep it or override it
        drops = []  # list of same column names for agent data and received data
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
        # self.obs_data = pd.merge(self.obs_data, data_to_concatenate, how='outer', on='index')
        self.obs_data = pd.concat([self.obs_data, data_to_concatenate], axis=1)

        # Decide how to manage NaN values if present

    def learning(self, nodes, parameters, non_doable, mod, bn=None, obs_data=None, edges=None):

        estimator = CausalLeaner(nodes=nodes, non_dobale=non_doable, edges=edges, env=bn, obs_data=obs_data)
        model, undirected_edges = estimator.learn(mod=mod, max_cond_vars=parameters['max_cond_vars'], do_size=parameters['do_size'], do_conf=parameters['do_conf'], ci_conf=parameters['ci_conf'])

        return model, undirected_edges

    # Check for incomplete nodes: for now this step is simulated, we add variables manually
    # def check_incomplete(self):
    #     # Example
    #     incomplete = ['T']
    #
    #     for node in incomplete:
    #         if node in self.nodes:
    #             self.incomplete.append(node)

    def print_structure(self):
        dot = draw(self.edges)
        dot.view(directory='tmp/tmp/')

    def build_request_msg(self, nodes_to_investigate: list, undirected_edges: list):
        # The message contains:
        #   - nodes with outliers values
        #   - nodes in undirected connections
        # In case of duplicates, eliminate them

        nodes_to_send = []
        if len(nodes_to_investigate) != 0:
            nodes_to_send.extend(nodes_to_investigate)
        if len(undirected_edges) != 0:
            nodes_to_send.extend(self.nodes_from_edges(undirected_edges))

        nodes_to_send = list(set(nodes_to_send))

        if len(nodes_to_send) != 0:
            non_doable = []
            for node in nodes_to_send:
                if node in self.non_doable:
                    non_doable.append(node)

            # Data are necessary for the chi-square
            # Example: Pow->W (non-doable->doable)
            # In this case we need data both for Pow and for W, because the chi-square compares the distributions
            obs_data = self.obs_data
            data_to_send = obs_data.drop(columns=[x for x in obs_data.columns if x not in nodes_to_send])

            # Build message
            msg = dict()
            msg['nodes'] = nodes_to_send
            msg['non_doable'] = non_doable
            msg['data'] = data_to_send
        else:
            return None

        return msg

    def build_response_msg(self, discovered_edges: list):
        msg = dict()

        non_doable = []
        nodes = self.nodes_from_edges(discovered_edges)
        for node in nodes:
            if node in self.non_doable:
                non_doable.append(node)

        if len(discovered_edges) != 0:
            msg['edges'] = discovered_edges
            msg['non_doable'] = non_doable
            return msg
        else:
            return None

    def read_request(self, request_msg):

        if request_msg:
            msg = request_msg
        else:
            return False  # Not going to learn

        # Check if all received nodes were already known: in this case it is useless to repeat the learning
        if all(item in self.nodes for item in msg['nodes']):
            print('Nodes already known, checking the previous learning results...')
            return False  # Not going to learn
        else:
            # Code for adding new nodes to structure
            # for node in msg['nodes']:
            #     if node not in self.nodes:
            #         self.add_node(node)
            #
            # for node in msg['non_doable']:
            #     if node not in self.non_doable:
            #         self.add_non_doable(node)
            #
            # # Concatenation of observational data
            # if msg['data'] is not None:
            #     self.concatenate_data(msg['data'])

            return True  # Going to learn

    def read_response(self, response):
        # We consider trusted the communication between agents, so we directly integrate the response
        # without repeat the learning

        if len(response) != 0:
            # Read nodes and add to structure
            new_nodes = self.nodes_from_edges(response['edges'])
            for node in new_nodes:
                if node not in self.nodes:
                    self.add_node(node)

            for node in response['non_doable']:
                if node not in self.non_doable:
                    self.add_non_doable(node)

            # Read edges and add to structure
            for t in response['edges']:
                if t not in self.edges:
                    self.add_edge(t)
        else:
            print('Empty response, nothing added')








