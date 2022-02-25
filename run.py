# This file contains a communication example between two agents
# In particular, agent 1 and agent 2, where agent 2 sends a request to agent 1

from agent import Agent
import networks
import time
from utils.config import parameters
import pandas as pd
from utils.drawing import draw


# Select only the results in which there are nodes required
def get_response_edges(predicted_edges, request_nodes):
    if predicted_edges is None or request_nodes is None:
        return 'None object in building edges for response'
    elif len(predicted_edges) == 0 or len(request_nodes) == 0:
        return []
    else:
        res = []
        for t in predicted_edges:
            if t[0] in request_nodes or t[1] in request_nodes:
                res.append(t)

        return res


# The function builds the list of new edges based on the new introduced nodes
def get_new_edges(nodes_to_investigate, nodes):
    if nodes_to_investigate is None or nodes is None:
        return 'None object in new edges building'
    elif len(nodes_to_investigate) == 0 and len(nodes) == 0:
        return []
    else:
        all_nodes = list(set(nodes_to_investigate + nodes))

        new_edges = []
        for node in nodes_to_investigate:
            for new in all_nodes:
                if new != node:
                    new_edges.append((new, node))
                    new_edges.append((node, new))

        return list(set(new_edges))


def nodes_from_edges(edges):
    nodes = []
    for edge in edges:
        nodes.append(edge[0])
        nodes.append(edge[1])
    return list(set(nodes))


def concatenate_data(old_data, new_data, override=True):
    # When a column is already present, decide if keep it or override it
    drops = []  # list of same column names for agent data and received data
    for col in new_data.columns:
        if col in old_data.columns:
            drops.append(col)

    if override:
        # override old data
        old_data.drop(columns=drops, inplace=True)
    else:
        # do not override old data
        new_data.drop(columns=drops, inplace=True)

    # Merge of data based on the id
    return pd.concat([old_data, new_data], axis=1)


# Divide network
network_1 = networks.get_network_from_nodes(['Pr', 'L', 'Pow', 'H', 'C', 'S'], False)
network_2 = networks.get_network_from_nodes(['CO', 'CO2', 'A', 'W', 'B', 'T', 'O'], False)

# This example works with simulated online intervention (call to icasa.simulate())
# The results shows the correct functioning of the algorithm
# network_1 = networks.get_network_from_nodes(['Pr', 'L', 'Pow'], False)
# network_2 = networks.get_network_from_nodes(['W', 'H', 'T'], False)

# Initialize agents
agent_1 = Agent(nodes=network_1['nodes'], non_doable=network_1['non_doable'], edges=network_1['edges'],  obs_data=network_1['dataset'])
agent_2 = Agent(nodes=network_2['nodes'], non_doable=network_2['non_doable'], edges=network_2['edges'], obs_data=network_2['dataset'])

start = time.time()

# ALGORITHM

# 1 - Offline Local learning
model_1, undirected_edges_1 = agent_1.learning(nodes=agent_1.nodes, non_doable=agent_1.non_doable, parameters=parameters, mod='offline', bn=agent_1.bn, obs_data=agent_1.obs_data)
agent_1.add_undirected_edges(undirected_edges_1)
dot = draw(model_1.edges())
dot.view(directory='tmp/1/')
model_2, undirected_edges_2 = agent_2.learning(nodes=agent_2.nodes, non_doable=agent_2.non_doable, parameters=parameters, mod='offline', bn=agent_2.bn, obs_data=agent_2.obs_data)
agent_2.add_undirected_edges(undirected_edges_2)
dot = draw(model_2.edges())
dot.view(directory='tmp/2/')

# 2 - Request
# For test reasons we choose the incomplete nodes manually
# In this case agent 2 wants to send a request message
nodes_to_investigate = ['H']
if len(nodes_to_investigate) != 0:
    msg = agent_2.build_request_msg(nodes_to_investigate, agent_2.undirected_edges)
    # print('Request message ', msg)
else:
    print('No communication')
    exit(0)

# 3 - Read request and build response
# The message is sent to agent 1 that reads it
# The reading does not modify the interior structure of agent 1
ret = agent_1.read_request(msg)


response = []
if ret:  # If we receive True, we repeat the learning

    # Partial learning as combination of new and agent: in particular we choose to not add the new nodes directly
    # to agent 1 structure, but treat them as a complete new learning
    new_edges = get_new_edges(msg['nodes'], agent_1.nodes)  # build edges list as a combination
    new_nodes = list(set(msg['nodes'] + agent_1.nodes))
    new_non_doable = list(set(msg['non_doable'] + agent_1.non_doable))
    if msg['data'] is not None:
        new_data = concatenate_data(agent_1.obs_data, msg['data'])
    else:
        new_data = agent_1.obs_data

    # Learn about new edges
    model, undirected_edges = agent_1.learning(nodes=new_nodes, non_doable=new_non_doable, parameters=parameters, mod='online', edges=new_edges, bn=None, obs_data=new_data)

    # The response contains the learnt edges
    response = agent_1.build_response_msg(model.edges())
else:  # If we receive False, we check edges already known by agent 1
    response = get_response_edges(agent_1.edges, nodes_to_investigate)
print('Response: ', response)

# Send response to agent 2

# 4 - Integration
# Read received response and update structure
agent_2.read_response(response)

end = time.time()
print('Time elapsed: ', (end-start), 's')

dot = draw(agent_2.edges)
dot.view(directory='tmp/3/')






