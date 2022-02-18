from ocik import Asia, Room, Circuit, RoomBase, RoomComplete, Test, BigRoom
from ocik import CausalLeaner
from ocik.network import BayesianNetwork
from graphviz import Digraph, Graph, Source
import pandas as pd
import time
import numpy as np
from drawing import draw, difference

import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\pakyr\\.conda\\envs\\bayesianEnv\\Library\\bin\\graphviz'

# Set true if the results are printable
printable = True
start = time.time()

# Room: class for the starting network

# room = Room()
# bn = room.get_network()
#
# # Generazione casuale di sample
# # obs_data_random = bn.sample(4000)
#
# obs_data = pd.read_csv('ocik/demo/store/room.csv')
# estimator = CausalLeaner(bn.nodes(), non_dobale=['L', 'T'], env=bn, obs_data=obs_data)
# model, undirected_edges = estimator.learn(max_cond_vars=4, do_size=100)

#######################################################################

# Base room: class for the base room

room = RoomBase()
bn = room.get_network()

obs_data_csv = pd.read_csv('ocik\\demo\\store\\room_base.csv')

estimator = CausalLeaner(bn.nodes(), non_dobale=['Pr', 'Pow', 'T'], env=bn, obs_data=obs_data_csv)
model, undirected_edges = estimator.learn(mod='online', max_cond_vars=4, do_size=2)

##########################################################################

# Complete room: class for the complete room

# room = RoomComplete()
# bn = room.get_network()
#
# obs_data_csv = pd.read_csv('ocik\\demo\\store\\room_complete.csv')
# obs_data_csv = obs_data_csv.reset_index()
#
# estimator = CausalLeaner(bn.nodes(), non_dobale=['Pr', 'Pow', 'O', 'T', 'CO', 'CO2'], env=bn, obs_data=obs_data_csv)
# model, undirected_edges = estimator.learn(max_cond_vars=4, do_size=100)

##########################################################################

# Test ######################################################

# room = Test()
# bn = room.get_network()
#
# obs_data_csv = pd.read_csv('ocik\\demo\\store\\room_complete.csv')
#
# estimator = CausalLeaner(bn.nodes(), non_dobale=['Pr', 'T', 'CO', 'CO2'], env=bn, obs_data=obs_data_csv)
# model, undirected_edges = estimator.learn(max_cond_vars=4, do_size=1)

#########################################################################

# Big Network: this class allows to build and test a random network with random CPD and specific number of nodes

# n_nodes = 20
# room = BigRoom(n_nodes=n_nodes)
# bn = room.get_network()
#
# # Since the network is created with numerical names, the printing methods do not work on it
# printable = False
#
#
# # Makes the nodes printable
# def masking(n_nodes, edges):
#     mask = {}
#     for i in range(n_nodes):
#         mask[i] = str(i)
#
#     n = []
#     for t in edges:
#         n.append((mask[t[0]], mask[t[1]]))
#     return n
#
#
# # For simplicity we assume not to have non-doable nodes
# estimator = CausalLeaner(bn.nodes(), non_dobale=[], env=bn, obs_data=None)
# model, undirected_edges = estimator.learn(max_cond_vars=4, do_size=1)
#
# masked_edges = masking(n_nodes, bn.edges())
# masked_model_edges = masking(n_nodes, model.edges)
#
# # Performances
# gt = bn.edges()
# pred = model.edges()
# recovered_edges = [ed for ed in pred if ed in gt]
# print('Recovered nodes:\t', len(recovered_edges), '/', len(gt), '\nRecovered rate:\t', len(recovered_edges)/len(gt)*100, '%')
# dot = difference(masked_edges, masked_model_edges)
# dot.view(directory='tmp/')
##########################################################################

end = time.time()
print('\nTime elapsed for the computation is: ', round(end-start, 2), ' s')

if printable:
    dot = difference(bn.edges(), model.edges())
    dot.view(directory='tmp/')
else:
    exit(200)

# Test con tracking attivo: produce un grafo per ogni step
# model, track = estimator.learn(max_cond_vars=4, do_size=100, trace=True, verbose=True)
# for i, edges in enumerate(track):
#     var = "order " + str(i) + " : " if i != len(track) - 1 else "final result after postprocessing:"
#     f = Digraph()
#     f.edges(edges)
#     print(f)

