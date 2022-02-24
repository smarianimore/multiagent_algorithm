from ocik.causal_leaner import CausalLeaner
import networks
from agent import Agent
from utils.config import parameters
from utils.drawing import draw

network = networks.get_network_from_nodes(['Pr', 'L', 'T', 'W', 'H', 'Pow'], False)
agent_1 = Agent(nodes=network['nodes'], non_doable=network['non_doable'], edges=network['edges'],  obs_data=network['dataset'])
model_1, undirected_edges_1 = agent_1.learning(nodes=agent_1.nodes, non_doable=agent_1.non_doable, parameters=parameters, mod='online', bn=agent_1.bn, obs_data=agent_1.obs_data)

dot = draw(model_1.edges())
dot.view(directory='tmp/3/')



