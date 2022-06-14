import os
from typing import Optional

import pandas as pd
from networkx import DiGraph

from ocik import CausalLeaner
from ocik.network import BayesianNetwork
from utils.cpt_estimator import ConditionalProbability

os.environ["PATH"] += "/usr/local/Cellar/graphviz/2.44.1/lib/graphviz"


# Class for an Agent of the environment
class Agent:
    def __init__(self, nodes: list[str], non_doable: list[str], gt_edges: list[tuple[str, str]],
                 obs_data: pd.DataFrame):
        """
        An agent learning a causal model

        Parameters
        ----------
        nodes : list[str]
            the list of nodes known to the agent
        non_doable : list[str]
            the list of non doable nodes
        gt_edges : list[tuple[str, str]]
            list of edges (ground truth, used for simulation of interventions)
        obs_data : pd.DataFrame
            all the observational data available for the 'nodes'
        """
        self.nodes = nodes
        self.gt_edges = gt_edges
        self.non_doable = non_doable
        self.obs_data = obs_data
        self.gt_cpt = ConditionalProbability(self.obs_data, self.gt_edges)  # DOC given ground truth build CPT
        self.gt_bn = self.create_gt_bn_net()
        self.undirected_edges = []
        self.incomplete = []

    def create_gt_bn_net(self) -> BayesianNetwork:
        """
        Creates the Bayesian Network (hence edges + CPT) of the ground truth edges

        Returns
        -------
        BayesianNetwork
            a 'BayesianNetwork' of the ground truth
        """
        bn = BayesianNetwork(self.gt_edges)
        for node in bn.nodes():
            parents = [edge[0] for edge in bn.edges() if node == edge[1]]
            arr = self.gt_cpt.get_node_prob(node)
            arr = [arr[1], arr[0]]  # invert the array for construction reasons
            bn.set_cpd(node, arr, parents)
        return bn

    def learning(self, parameters: dict[str: int], mod: str, edges: list[tuple[str, str]] = None) \
            -> tuple[DiGraph, list[tuple[str, str]]]:
        """

        Parameters
        ----------
        parameters :
            a dictionary storing the learning parameters
        mod :
            'offline' to learn from data and simulate interventions, 'online' to intervene on running iCasa simulation
        edges :
            QUESTION ???

        Returns
        -------
        tuple[DiGraph, list[tuple[str, str]]]
            the learnt model and the list of undirected edges found
        """
        estimator = CausalLeaner(nodes=self.nodes,
                                 non_dobale=self.non_doable,
                                 edges=edges,
                                 env=self.gt_bn,  # DOC used to simulate interventions and evaluate performance
                                 obs_data=self.obs_data)
        model, undirected_edges = estimator.learn(mod=mod,
                                                  max_cond_vars=parameters['max_cond_vars'],
                                                  do_size=parameters['do_size'],
                                                  do_conf=parameters['do_conf'],
                                                  ci_conf=parameters['ci_conf'])

        return model, undirected_edges

    # Get non-duplicate nodes list from edges
    @staticmethod
    def nodes_from_edges(edges):
        nodes = []
        for edge in edges:
            nodes.append(edge[0])
            nodes.append(edge[1])
        return list(set(nodes))

    def build_request_msg(self, frontier: list, undirected_edges: list) -> Optional[dict]:
        """
        Builds the help message to ask help to other agents

        Parameters
        ----------
        frontier :
            nodes with "outlier values" (taken as given, atm)
        undirected_edges :
            edges whose direction should be established yet

        Returns
        -------
        dict
            a dictionary of message data (nodes to investigate, which are non doable, known observational data)
        """
        nodes_to_send = []
        if len(frontier) != 0:
            nodes_to_send.extend(frontier)
        if len(undirected_edges) != 0:
            nodes_to_send.extend(self.nodes_from_edges(undirected_edges))
        nodes_to_send = list(set(nodes_to_send))

        if len(nodes_to_send) != 0:
            non_doable = [node for node in nodes_to_send if node in self.non_doable]
            # Data are necessary for the chi-square
            # Example: Pow->W (non-doable->doable)
            # In this case we need data both for Pow and for W, because the chi-square compares the distributions
            # NB defaults to 'inplace=False' hence no column is removed from original dataframe
            data_to_send = self.obs_data.drop(columns=[x for x in self.obs_data.columns if x not in nodes_to_send])

            msg = dict()
            msg['nodes'] = nodes_to_send
            msg['non_doable'] = non_doable
            msg['data'] = data_to_send
        else:
            return None

        return msg

    def build_response_msg(self, discovered_edges: list[tuple[str, str]]) -> Optional[dict]:
        """

        Parameters
        ----------
        discovered_edges : list[tuple[str, str]]
            the list of edges discovered by the replying agent

        Returns
        -------
        dict
            a dictionary of message data (discovered edges, non doable nodes)
        """
        msg = dict()

        nodes = self.nodes_from_edges(discovered_edges)
        non_doable = [node for node in nodes if node in self.non_doable]
        if len(discovered_edges) != 0:
            msg['edges'] = discovered_edges
            msg['non_doable'] = non_doable
            return msg
        else:
            return None

    def read_request(self, request_msg: dict) -> bool:
        """

        Parameters
        ----------
        request_msg : dict
            a dictionary of message data (nodes to investigate, which are non doable, known observational data)

        Returns
        -------
        bool
            whether all the received nodes are already known, hence no further learning is to be done locally
        """
        if request_msg:
            msg = request_msg
        else:
            return False  # Not going to learn

        # Check if all received nodes were already known: in this case it is useless to repeat the learning
        if all(item in self.nodes for item in msg['nodes']):
            print('Nodes already known, checking the previous learning results...')
            return False  # Not going to learn
        else:
            # QUESTION we do nothing?
            # Code for adding new nodes to existing structure before to make incremental learning
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

    def add_node(self, node):
        self.nodes.append(node)

    def add_non_doable(self, node):
        self.non_doable.append(node)

    def add_edge(self, edge):
        self.gt_edges.append(edge)

    def read_response(self, response: dict):
        """

        Parameters
        ----------
        response : dict
            a dictionary of message data (discovered edges, non doable nodes)
        """
        if len(response) != 0:
            new_nodes = self.nodes_from_edges(response['edges'])
            for node in new_nodes:
                if node not in self.nodes:
                    self.add_node(node)
            for node in response['non_doable']:
                if node not in self.non_doable:
                    self.add_non_doable(node)
            for t in response['edges']:
                if t not in self.gt_edges:
                    self.add_edge(t)
        else:
            print('Empty response, nothing added')

    # replace list of edges
    def replace_edges(self, learned_edges):
        self.gt_edges.clear()
        self.gt_edges = learned_edges

    def add_undirected_edges(self, undirected_edges):
        for edge in undirected_edges:
            self.undirected_edges.append(edge)

    # FIXME unused apparently
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

        # TODO Decide how to manage NaN values if present
