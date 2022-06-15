import datetime
import os
import time

from networkx import DiGraph

from agent import Agent
from networks import create_gt_net_skel
from utils.config import resp_time
from utils.drawing import difference
from utils.logging import append_to_report

os.environ["PATH"] += "/usr/local/Cellar/graphviz/2.44.1/lib/graphviz"  # TODO put in config file


def test(params: dict[str: int], network: dict[str: list], mod: str) -> tuple[DiGraph, float]:
    """
    Tests one agent's learning ability on the given 'network', with the given 'params', in the given 'mod'

    Parameters
    ----------
    params : dict[str: int]
        a dictionary storing the learning parameters
    network : dict[str: list]
        a dictionary storing the learning parameters
    mod : str
        'offline' to learn from data and simulate interventions, 'online' to intervene on running iCasa simulation

    Returns
    -------
    tuple[DiGraph, float]
        the learnt model and the learning time
    """
    agent = Agent(nodes=network['nodes'],  # TODO put keys in config file
                  non_doable=network['non_doable'],
                  gt_edges=network['edges'],
                  obs_data=network['dataset'])
    start = time.time()

    model, undirected_edges = agent.learning(parameters=params, mod=mod)

    end = time.time()
    elapsed = (end - start)

    return model, elapsed


def report_results(parameters: dict[str: int], network: dict[str: list], model: DiGraph, elapsed_time: float,
                   output_name: str, mod: str, directory: str):
    """
    Reports learning results for 'model' in 'output_name' file within folder 'directory'.

    Parameters
    ----------
    parameters : dict[str: int]
        a dictionary storing the learning parameters
    network : dict[str: list]
        a dictionary representing the network whose causal model should be learnt
    model : DiGraph
        the model actually learnt
    elapsed_time : float
        the model actually learnt
    output_name : str
        the filename where to write results
    mod : str
        'offline' to learn from data and simulate interventions, 'online' to intervene on running iCasa simulation
    directory : str
        the folder where to store the 'output_name' file

    """
    # Parameters
    append_to_report('\nParameters:\n')
    for par in parameters:
        append_to_report(f'\t{par} = {parameters[par]}\n')
    if mod == 'online':
        append_to_report(f'\tresp_time = {resp_time}\n')

    gt_nodes = network['nodes']
    gt_edges = network['edges']
    pred_nodes = model.nodes
    pred_edges = model.edges
    n_gt = len(gt_edges)

    # Draw graph
    dot, spurious_edges, missed_edges, recovered_edges = difference(gt_edges, pred_edges, stat=True)
    dot.view(directory=directory, filename=output_name)
    n_spur = len(spurious_edges)
    n_missed = len(missed_edges)
    n_recovered = len(recovered_edges)

    # Report graph stats and time
    gt_net = f"\nGround truth:\n\t{len(gt_nodes)} nodes: {gt_nodes}\n\t{n_gt} edges: {gt_edges}"
    pred_net = f"\nPredicted:\n\t{len(pred_nodes)} nodes: {pred_nodes}\n\t{len(pred_edges)} edges: {pred_edges}"
    missed = f"\nMissed:\t ({n_missed}) \t {missed_edges}"
    spur = f"\nSpurious:\t ({n_spur}) \t {spurious_edges}\n"
    comp_time = f"\nLearning time: {elapsed_time} s\n"
    results = gt_net + pred_net + missed + spur + comp_time
    append_to_report(results)

    # Performance
    recover_rate = f'\nRecover rate: {(n_recovered / n_gt) * 100} %'
    append_to_report(recover_rate)
    missed_rate = f'\nMiss rate: {(n_missed / n_gt) * 100} %\n'
    append_to_report(missed_rate)


def do_tests(params: dict[str: int], nodes: list[str], notes: str, directory: str, mod: str = 'offline'):
    """
    Tests learning of a single agent.
    In order to run the test you have to
      1. Configure the network you want to test ('nodes')
      2. Choose the learning modality ('mod')
      3. Set the parameters for the learning algorithm ('params')
      4. Set the directory where you want the resulting graph ('directory')
      5. Optional: add some notes for the report ('notes')

    Parameters
    ----------
    params : dict[str: int]
        a dictionary storing the learning parameters
    nodes : list[str]
        the list of nodes whose causal model should be learnt
    notes : str
        optional notes to write on report file
    directory : str
        where to save output reports and graphs
    mod : str
        'offline' to learn from data and simulate interventions, 'online' to intervene on running iCasa simulation

    """
    # t0 = get_network_from_nodes(['H', 'T', 'C'], False)
    # t1 = get_network_from_nodes(['Pr', 'L', 'Pow', 'H', 'W', 'T', 'O'], False)
    # t2 = get_network_from_nodes(['Pr', 'L', 'Pow', 'H', 'C', 'W', 'B', 'T', 'O'], False)
    # t3 = get_network_from_nodes(['Pr', 'L', 'Pow', 'S', 'H', 'C', 'W', 'B', 'T', 'O'], False)
    # t4 = get_network_from_nodes(['Pr', 'L', 'Pow', 'S', 'H', 'C', 'CO', 'CO2', 'A', 'W', 'B', 'T', 'O'], False)
    t = create_gt_net_skel(nodes, False)  # DOC exploits ground truth to generate network description (it is a dict)
    tests = [t]

    # Initialization
    total_time = 0
    append_to_report(f'Timestamp: {datetime.datetime.now()}\n')
    append_to_report(f'\nLearning mode:\n\t{mod}\n\t1 agent\n')
    # DOC Leave some comments to clarify the possible meaning of a test
    append_to_report(f'\nNotes:\n\t{notes}\n')

    # Testing process
    # TODO refactor nodes to be a list of tuples ('name', [nodes]) and use 'name' for output filename
    for i, net in enumerate(tests):  # 'enumerate' generates indexes
        output_name = f'network_{datetime.datetime.now()}_' + str(i + 1)

        model, elapsed = test(params, net, mod)

        elapsed_time = round(elapsed, 2)
        total_time += elapsed_time
        append_to_report(f'\n----- Test {i + 1}:\n')
        print(f'\n----- Test {i + 1}:\n')
        report_results(params, net, model, elapsed_time, output_name, mod, directory)
        append_to_report(f'\n---------------------\n')
        print(f'\n---------------------\n')

    print('Tests time: ', total_time, ' s')
    append_to_report(f'\nTests time: {total_time} s')


# TODO put in config file
# DOC learning params, to be set empirically TODO grid-search procedure
PARAMS = {'max_cond_vars': 4,
          'do_size': 500,
          'do_conf': 0.9,
          'ci_conf': 0.4}
# DOC the network to learn
NODES = ['Pr', 'L', 'Pow', 'S', 'H', 'C', 'CO', 'CO2', 'A', 'W', 'B', 'T', 'O']
# DOC written on output file
NOTES = "testing reproducibility 2"
# DOC where to put output files
DIR = "output/reproducibility/"
# DOC intervention mode
MOD = 'offline'

if __name__ == "__main__":
    do_tests(PARAMS, NODES, NOTES, DIR, MOD)

    # Call n times when learning online, then pick best results
    # for n in range(5):
    #     single_agent_procedure()
