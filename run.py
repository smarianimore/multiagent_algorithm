from agent import Agent
import networks
import time
from utils.config import parameters


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


# Get networks definitions
network_1 = networks.get_network_from_nodes(['Pr', 'L', 'Pow', 'H', 'C', 'S'], False)
network_2 = networks.get_network_from_nodes(['CO', 'CO2', 'A', 'W', 'B', 'T', 'O'], False)

# Initialize agents
agent_1 = Agent(nodes=network_1['nodes'], non_doable=network_1['non_doable'], edges=network_1['edges'],  obs_data=network_1['dataset'])
agent_2 = Agent(nodes=network_2['nodes'], non_doable=network_2['non_doable'], edges=network_2['edges'], obs_data=network_2['dataset'])

start = time.time()

# ALGORITHM

# 1 - Offline Local learning
agent_1.learning(parameters=parameters, mod='offline')
# dot = draw(model_1.edges())
# dot.view(directory='tmp/1/')
agent_2.learning(parameters=parameters, mod='offline')
# dot = draw(model_2.edges())
# dot.view(directory='tmp/2/')

# 2 - Request
# For test reasons we choose the incomplete nodes manually
# In this case agent 2 wants to send a request message
nodes_to_investigate = ['T']
msg = agent_2.build_request_msg(nodes_to_investigate, agent_2.undirected_edges)
# print('Request message ', msg)

# 3 - Read request and build response
# The message is sent to agent 1 that reads it
ret = agent_1.read_request(msg)

response = []
if ret:  # If we receive True, we repeat the learning
    # Fill in with new edges to test
    new_edges = get_new_edges(nodes_to_investigate, agent_1.nodes)
    # Learn about new edges
    agent_1.learning(parameters=parameters, mod='online', edges=new_edges)
    # Build response with matching between agent 1 edges and requested nodes
    response = get_response_edges(agent_1.edges, nodes_to_investigate)
else:  # If we receive False, we check edges already known by agent 1
    response = get_response_edges(agent_1.edges, nodes_to_investigate)
print('Response: ', response)

# Send response to agent 2

# 4 - Integration
# Read received response and update structure
agent_2.read_response(response)

end = time.time()
print('Time elapsed: ', (end-start), 's')

# dot = draw(agent_2.edges)
# dot.view(directory='tmp/3/')






