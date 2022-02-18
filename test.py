from ocik import Asia, Room, Circuit, RoomBase, RoomMiddle1, RoomMiddle2, RoomComplete
from ocik import CausalLeaner
from graphviz import Digraph, Graph
import pandas as pd
import time
import datetime

import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\pakyr\\.conda\\envs\\bayesianEnv\\Library\\bin\\graphviz'

# Report path
REPORT = 'tmp.txt'


def draw(edge, directed=True):
    dot = Digraph(graph_attr={'rankdir': 'LR'}) if directed else Graph()
    dot.edges(edge)
    return dot


def difference(gt, pred):
    f = Digraph(graph_attr={'rankdir': 'LR'})
    new_edges = [ed for ed in pred if ed not in gt]
    f.attr('edge', color='blue')
    f.edges(new_edges)

    missed_edges = [ed for ed in gt if ed not in pred]
    f.attr('edge', color='red')
    f.edges(missed_edges)

    recovered_edges = [ed for ed in pred if ed in gt]
    f.attr('edge', color='green')
    f.edges(recovered_edges)
    return f, new_edges, missed_edges, recovered_edges


def test(network, non_doable, obs_data, params):
    room = network
    bn = room.get_network()

    obs_data_csv = pd.read_csv(f'ocik\\demo\\store\\test\\{obs_data}.csv')
    estimator = CausalLeaner(bn.nodes(), non_dobale=non_doable, env=bn, obs_data=obs_data_csv)
    model = estimator.learn(max_cond_vars=params['max_cond_vars'], do_conf=params['do_conf'], ci_conf=params['ci_conf'], do_size=params['do_size'])

    return bn, model


def print_results(bn, model, params, elapsed_time, output_name):
    # Parameters
    write_on_report('\n\nParameters:')
    for par in params:
        write_on_report(f'\n{par} = {params[par]}')

    # Topology and computational time
    results = f'\n\nTopology: {bn.G} \nNodes: {bn.nodes()} \nEdges: {bn.edges()} \nComputational time: {elapsed_time} s'
    print(results)
    write_on_report(results)

    # Saving results
    dot, new_edges, missed_edges, recovered_edges = difference(bn.edges(), model.edges())
    dot.view(directory=f'tmp/test/{output_name}')

    # Edge statistics
    n_edges = len(bn.edges())  # Ground-truth number of edges
    new_edges = len(new_edges)
    missed_edges = len(missed_edges)
    recovered_edges = len(recovered_edges)

    # Performance measure
    edge_results = f'\nNew edges: {new_edges} \nMissed edges: {missed_edges} \nRecovered edges: {recovered_edges}'
    write_on_report(edge_results)
    recover_rate = f'\nRecover rate: {(recovered_edges / n_edges) * 100} %'
    write_on_report(recover_rate)
    missed_rate = f'\nMissed rate: {(missed_edges / n_edges) * 100} %'
    write_on_report(missed_rate)


def write_on_report(text, file=REPORT):
    with open(file, 'a') as f:
        f.write(text)


if __name__ == "__main__":
    # In order to run the test you have to
    # 1. Configure the networks you want to test and include it in the tests array
    # 2. Optional: add some notes for the report
    # 3. Set the parameters for the learning algorithm

    # Definitions of the networks
    t1 = [RoomBase(), ['Pr', 'Pow', 'T'], 'network']
    t2 = [RoomMiddle1(), ['Pr', 'Pow', 'T'], 'network']
    t3 = [RoomMiddle2(), ['Pr', 'Pow', 'T'], 'network']
    t4 = [RoomComplete(), ['Pr', 'Pow', 'T', 'CO', 'CO2'], 'network']

    # Array containing the networks to be tested
    tests = [t1, t2, t3, t4]

    # Initialization
    total_time = 0
    write_on_report(f'\n\n#### {datetime.datetime.now()}')

    # Leave some comments to clarify the possible meaning of a test
    # For example: "Incremental do_size" means that the test is made to verify do_size variable effects
    notes = "search for the best recover rate"
    write_on_report(f'\nNotes:\t{notes}')

    max_cond_vars = 4  # max number of variables to condition on (seems not to make any difference)
    do_conf = 0.9  # confidence on do operation
    ci_conf = 0.5  # confidence level for chi-square test on non-doable nodes
    do_size = 10  # number of do operation for each evidence
    params = {'max_cond_vars': max_cond_vars,
              'do_conf': do_conf,
              'ci_conf': ci_conf,
              'do_size': do_size}

    # Testing process
    for i, t in enumerate(tests):
        instance = t[0]
        non_doable = t[1]
        obs_data = t[2]
        output_name = 'network_' + str(i + 1)

        start = time.time()
        bn, model = test(instance, non_doable, obs_data, params)
        end = time.time()

        elapsed_time = round((end-start), 2)
        total_time += elapsed_time
        write_on_report(f'\n\n## Test {i + 1}')
        print_results(bn, model, params, elapsed_time, output_name)

    print('Elapsed total time: ', total_time, ' s')
    write_on_report(f'\n\nElapsed total time: {total_time} s')
