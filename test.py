from utils.drawing import draw, difference
from networks import get_network_from_nodes
from utils.config import parameters, resp_time
from agent import Agent
from run import run
import datetime
import time


import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\pakyr\\.conda\\envs\\bayesianEnv\\Library\\bin\\graphviz'

# Report path
REPORT = 'single_agent.txt'


def double_agent_test(network_1, network_2, nodes_to_investigate):
    agent_1 = Agent(nodes=network_1['nodes'], non_doable=network_1['non_doable'], edges=network_1['edges'],
                    obs_data=network_1['dataset'])
    agent_2 = Agent(nodes=network_2['nodes'], non_doable=network_2['non_doable'], edges=network_2['edges'],
                    obs_data=network_2['dataset'])
    model_1, model_2, new_edges, elapsed = run(agent_1, agent_2, nodes_to_investigate)

    return model_1, model_2, new_edges, elapsed


# single agent -> complete learning
def single_agent_test(network, mod):
    agent = Agent(nodes=network['nodes'], non_doable=network['non_doable'], edges=network['edges'],
                    obs_data=network['dataset'])

    start = time.time()
    model, undirected_edges = agent.learning(nodes=agent.nodes, non_doable=agent.non_doable,
                                             parameters=parameters, mod=mod, bn=agent.bn,
                                             obs_data=agent.obs_data)
    end = time.time()
    elapsed = (end - start)

    return model, elapsed


def print_results_single_agent(network, model, elapsed_time, output_name, mod):
    # Parameters
    write_on_report('\n\nParameters:')
    for par in parameters:
        write_on_report(f'\n{par} = {parameters[par]}')
    if mod == 'online':
        write_on_report(f'\nresp_time = {resp_time}')

    # Topology and computational time
    gt_nodes = network['nodes']
    gt_edges = network['edges']
    pred_nodes = model.nodes
    pred_edges = model.edges

    # Print figures
    dot, new_edges, missed_edges, recovered_edges = difference(gt_edges, pred_edges, stat=True)
    dot.view(directory=f'tmp/test/')

    gt_net = f"\n\nGround-truth \nNodes: {gt_nodes}\tlen={len(gt_nodes)} \nEdges: {gt_edges}\tlen={len(gt_edges)}"
    pred_net = f"\nPredicted \nNodes: {pred_nodes}\tlen={len(pred_nodes)} \nEdges: {pred_edges}\tlen={len(pred_edges)}"
    missed = f"\nMissed edges: {missed_edges}"
    comp_time = f"\nComputational time: {elapsed_time} s"
    results = gt_net + pred_net + missed + comp_time
    print(results)
    write_on_report(results)

    # Edge statistics
    n_edges = len(gt_edges)  # Ground-truth number of edges
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


def single_agent_procedure():
    # PROCEDURE FOR SINGLE AGENT TEST
    # In order to run the test you have to
    # 1. Configure the networks you want to test and include it in the tests array
    # 2. Choose the learning modality (online or offline)
    # 3. Set the parameters for the learning algorithm from config file
    # 4. Optional: add some notes for the report

    # Definitions of the networks
    t0 = get_network_from_nodes(['H', 'T', 'C'], False)
    t1 = get_network_from_nodes(['Pr', 'L', 'Pow', 'H', 'W', 'T', 'O'], False)
    t2 = get_network_from_nodes(['Pr', 'L', 'Pow', 'H', 'C', 'W', 'B', 'T', 'O'], False)
    t3 = get_network_from_nodes(['Pr', 'L', 'Pow', 'S', 'H', 'C', 'W', 'B', 'T', 'O'], False)
    t4 = get_network_from_nodes(['Pr', 'L', 'Pow', 'S', 'H', 'C', 'CO', 'CO2', 'A', 'W', 'B', 'T', 'O'], False)

    # Array containing the networks to be tested
    tests = [t4]

    # Initialization
    total_time = 0
    write_on_report(f'\n\n#### {datetime.datetime.now()}')

    # Choose learning modality
    n_agents = 1
    mod = 'online'
    write_on_report(f'\nLearning modality:\t{mod}\t{n_agents} agent')

    # Leave some comments to clarify the possible meaning of a test
    # For example: "Incremental do_size" means that the test is made to verify do_size variable effects
    notes = "using the same parameters as multi-agent"
    write_on_report(f'\nNotes:\t{notes}')

    # Testing process
    for i, net in enumerate(tests):
        network = net
        output_name = 'network_' + str(i + 1)

        model, elapsed = single_agent_test(network, mod)

        elapsed_time = round(elapsed, 2)
        total_time += elapsed_time
        write_on_report(f'\n\n## Test {i + 1}')
        print_results_single_agent(network, model, elapsed_time, output_name, mod)

    print('Elapsed total time: ', total_time, ' s')
    write_on_report(f'\n\nElapsed total time: {total_time} s')


def double_agent_procedure():
    network_1 = get_network_from_nodes(['T', 'H', 'C'], False)
    network_2 = get_network_from_nodes([], False)
    nodes_to_investigate = []

    # Initialization
    total_time = 0
    write_on_report(f'\n\n#### {datetime.datetime.now()}')

    # Choose learning modality
    n_agents = 2
    mod = 'mixed'
    write_on_report(f'\nLearning modality:\t{mod}\t{n_agents} agent')

    # Leave some comments to clarify the possible meaning of a test
    # For example: "Incremental do_size" means that the test is made to verify do_size variable effects
    notes = "forming test script"
    write_on_report(f'\nNotes:\t{notes}')

    model_1, model_2, new_edges, elapsed = double_agent_test(network_1, network_2, nodes_to_investigate)

    # model_1 contiene la struttura dell'agente che ha ricevuto richiesta
    # model_2 contiene la struttura dell'agente che ha fatto richiesta
    # new_edges contiene gli edges scoperti da chi ha ricevuto richiesta
    # (model_2.edges - new_edges) è la struttura dell'agente richiedente prima della richesta

    # Con questo tipo di learning si può confrontare se effettivamente la comunicazione aiuta nell'apprendimento
    # di alcune connessioni. Es: comunicando T, si riceve H->T, ma nella realtà esiste anche C->T. Questo è indice
    # del fatto che, in questo caso, si è appreso la metà 1 edge su 2.
    # In teoria ci si può riferire sempre alla ground-truth per analizzare i risultati.
    # Indicare quante nuove connessioni sono state apprese:
    # - rispetto a quelle che l'agente conosceva (new_edges)
    # - rispetto a quante ne esistono realmente (new_edges|gt_edges)
    # Esempio:
    # network_1 = ['Pr', 'L', 'Pow', 'H', 'C', 'S']
    # network_2 = ['CO', 'CO2', 'A', 'W', 'B', 'T', 'O']
    # nodes_to_investigate = ['T']
    # L'agente 2 sa che esistono O->T e W->T, ma non conosce H->T e C->T
    # comunicando con l'agente 1 scopre H->T
    # in questo caso l'algoritmo ha scoperto una nuova connessione rispetto a quelle che l'agente 2 conosceva,
    # ma solamente 1/2 di quelle che avrebbe potuto scoprire, infatti manca C->T.


if __name__ == "__main__":
    single_agent_procedure()

    # Call n times when learning online, then pick best results
    # for n in range(5):
    #     single_agent_procedure()

