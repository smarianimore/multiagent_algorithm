from ocik.causal_leaner import CausalLeaner
import networks
from agent import Agent
from utils.config import parameters
from utils.drawing import draw, difference
import time


def learning(agent, mod):
    start = time.time()
    model, undirected_edges = agent.learning(nodes=agent.nodes, non_doable=agent.non_doable,
                                                   parameters=parameters, mod=mod, bn=agent.bn,
                                                   obs_data=agent.obs_data)
    end = time.time()

    print('Elapsed time: ', end - start, 's')
    dot = difference(network['edges'], model.edges())
    dot.view(directory='tmp/3/')


if __name__ == '__main__':
    complete = ['Pr', 'L', 'Pow', 'H', 'C', 'S', 'CO', 'CO2', 'A', 'W', 'B', 'T', 'O']
    partial = ['H', 'C', 'T']

    network = networks.get_network_from_nodes(complete, False)

    agent = Agent(nodes=network['nodes'], non_doable=network['non_doable'], edges=network['edges'],
                    obs_data=network['dataset'])

    learning(agent, "online")







