# This file contains a communication example between two agents
# In particular, agent 1 and agent 2, where agent 2 sends a request to agent 1
# The request contains nodes that the agent 2 finds as incomplete or nodes from undirected edges
# If there are not such nodes, no message will be sent

from utils.drawing import draw, difference
from utils.config import parameters
from agent import Agent
import pandas as pd
import threading
import networks
import datetime
import time

# TODO put in config file
from utils.logging import append_to_report

REPORT = 'multi_agent_' + str(datetime.datetime.now()) + '.txt'

online_parameters = {
    'max_cond_vars': 4,
    'do_size': 3,
    'do_conf': 0.6,
    'ci_conf': 0.1
}

agent_1_parameters = {
    'max_cond_vars': 4,
    'do_size': 500,
    'do_conf': 0.9,
    'ci_conf': 0.4
}

agent_2_parameters = {
    'max_cond_vars': 4,
    'do_size': 500,
    'do_conf': 0.9,
    'ci_conf': 0.4
}


# Select only the edges in which there are nodes required
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


# Retrieve nodes from edges list
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


def write_results(gt, pred, elapsed_time, description):
    append_to_report(f"\n{description}", file=REPORT)

    gt_nodes = nodes_from_edges(gt)
    gt_edges = pred
    pred_nodes = nodes_from_edges(pred)
    pred_edges = pred

    # Print figures
    dot, new_edges, missed_edges, recovered_edges = difference(gt, pred, stat=True)
    # dot.view(directory=f'tmp/test/{output_name}')

    gt_net = f"\nGround-truth \nNodes: {gt_nodes}\tlen={len(gt_nodes)} \nEdges: {gt_edges}\tlen={len(gt_edges)}"
    pred_net = f"\nPredicted \nNodes: {pred_nodes}\tlen={len(pred_nodes)} \nEdges: {pred_edges}\tlen={len(pred_edges)}"
    missed = f"\nMissed edges: {missed_edges}"
    comp_time = f"\nComputational time: {elapsed_time} s"
    results = gt_net + pred_net + missed + comp_time
    append_to_report(results, file=REPORT)

    # Edge statistics
    n_edges = len(gt)  # Ground-truth number of edges
    new_edges = len(new_edges)
    missed_edges = len(missed_edges)
    recovered_edges = len(recovered_edges)

    # Performance measure
    edge_results = f'\nNew edges: {new_edges} \nMissed edges: {missed_edges} \nRecovered edges: {recovered_edges}'
    append_to_report(edge_results, file=REPORT)
    recover_rate = f'\nRecover rate: {(recovered_edges / n_edges) * 100} %'
    append_to_report(recover_rate, file=REPORT)
    missed_rate = f'\nMissed rate: {(missed_edges / n_edges) * 100} %'
    append_to_report(missed_rate, file=REPORT)


# Double-agent Learning
def run(agent_1, agent_2, gt_1, gt_2, gt_3, nodes_to_investigate, timestamp):
    start = time.time()
    # 1 - Offline Local learning (Agent 1 and Agent 2)
    model_1, undirected_edges_1 = agent_1.learning(nodes=agent_1.nodes, non_doable=agent_1.non_doable,
                                                   parameters=agent_1_parameters, mod='offline', bn=agent_1.gt_bn,
                                                   obs_data=agent_1.obs_data)
    time_1 = (time.time() - start)
    agent_1.replace_edges(list(model_1.edges()))
    agent_1.add_undirected_edges(undirected_edges_1)
    dot = difference(gt_1, model_1.edges())  # compare gt and pred
    dot.view(filename='1', directory=f'tmp/{timestamp}/')
    write_results(gt_1, model_1.edges(), time_1, "\nAgent 1 network")

    start = time.time()
    model_2, undirected_edges_2 = agent_2.learning(nodes=agent_2.nodes, non_doable=agent_2.non_doable,
                                                   parameters=agent_2_parameters, mod='offline', bn=agent_2.gt_bn,
                                                   obs_data=agent_2.obs_data)
    time_2 = (time.time() - start)
    agent_2.replace_edges(list(model_2.edges()))
    agent_2.add_undirected_edges(undirected_edges_2)
    dot = difference(gt_2, model_2.edges())  # compare gt and pred
    dot.view(filename='2', directory=f'tmp/{timestamp}/')
    write_results(gt_2, model_2.edges(), time_2, "\nAgent 2 network")

    # 2 - Request (Agent 2)
    if len(nodes_to_investigate) != 0 or len(agent_2.undirected_edges) != 0:
        # Nodes in the message are the ones indicated from the user (still manually) and the ones
        # from undirected edges discovered by the local learning algorithm
        msg = agent_2.build_request_msg(nodes_to_investigate, agent_2.undirected_edges)
        append_to_report(f'\n\nUndirected edges: {agent_2.undirected_edges}', file=REPORT)
        append_to_report(f'\nNodes to investigate: {nodes_to_investigate}', file=REPORT)
        print('Request message ', msg)
    else:
        print('No communication, ending.')
        exit(0)

    # 3 - Read request and build response (Agent 1)
    # The message is sent to agent 1 that reads it, without add any new nodes to agent 1 structure
    ret = agent_1.read_request(msg)

    response = []
    if ret:  # If we receive True, we do a partial learning investigating new nodes
        # Partial learning as combination of new and agent knowledge (no modification of agent 1 network)
        new_edges = get_new_edges(msg['nodes'], agent_1.nodes)
        new_nodes = list(set(msg['nodes'] + agent_1.nodes))
        new_non_doable = list(set(msg['non_doable'] + agent_1.non_doable))
        if msg['data'] is not None:
            new_data = concatenate_data(agent_1.obs_data, msg['data'])
        else:
            new_data = agent_1.obs_data

        start = time.time()
        # Learn about new edges
        model, undirected_edges = agent_1.learning(nodes=new_nodes, non_doable=new_non_doable,
                                                   parameters=online_parameters,
                                                   mod='online', edges=new_edges, bn=None, obs_data=new_data)
        time_3 = (time.time() - start)
        # The response contains the learnt edges
        response = agent_1.build_response_msg(model.edges())
        append_to_report(f'\nOnline predicted edges: {model.edges()}', file=REPORT)
    else:  # If we receive False, we check edges already known by agent 1
        response = get_response_edges(agent_1.edges, nodes_to_investigate)
    print('Response: ', response)

    # Send response to agent 2

    # 4 - Integration (Agent 2)
    # Read received response and update structure
    agent_2.read_response(response)

    end = time.time()
    elapsed_time = (end - start)
    print('Time elapsed: ', elapsed_time, 's')

    # difference function needs ground-truth network to visualize results
    # dot = difference(agent_2.edges, model_2.edges())
    dot = difference(gt_3, agent_2.edges)
    dot.view(filename='3', directory=f'tmp/{timestamp}/')
    write_results(gt_3, agent_2.edges, time_3, "\nAgent 2 network after request")

    return model_1, model_2, response['edges'], elapsed_time


if __name__ == '__main__':
    now = datetime.datetime.now()
    timestamp = datetime.datetime.timestamp(now)
    append_to_report(f'\n\n#### {now}', file=REPORT)
    append_to_report(f'\ntimestamp: {timestamp}', file=REPORT)
    append_to_report(f'\nMulti-agent learning', file=REPORT)

    notes = ""
    append_to_report(f'\nNotes:\t{notes}', file=REPORT)

    append_to_report('\n\nOffline agent 1 parameters:', file=REPORT)
    for par in agent_1_parameters:
        append_to_report(f'\n{par} = {agent_1_parameters[par]}', file=REPORT)

    append_to_report('\n\nOffline agent 2 parameters:', file=REPORT)
    for par in agent_2_parameters:
        append_to_report(f'\n{par} = {agent_2_parameters[par]}', file=REPORT)

    append_to_report('\n\nOnline parameters:', file=REPORT)
    for par in online_parameters:
        append_to_report(f'\n{par} = {online_parameters[par]}', file=REPORT)

    # Choose how to divide the network
    network_1 = networks.create_gt_net_skel(['Pr', 'L', 'Pow', 'S', 'H', 'C'], False)
    network_2 = networks.create_gt_net_skel(['W', 'O', 'T', 'B', 'A', 'CO', 'CO2'], False)

    # Initialize agents
    agent_1 = Agent(nodes=network_1['nodes'], non_doable=network_1['non_doable'], edges=network_1['edges'],
                    obs_data=network_1['dataset'])
    agent_2 = Agent(nodes=network_2['nodes'], non_doable=network_2['non_doable'], edges=network_2['edges'],
                    obs_data=network_2['dataset'])

    # List the nodes to send in the message from agent 2 to agent 1
    # So far, this array is managed manually, but the idea is to build an automatic method to recognize the nodes to
    # investigate from outliers values
    nodes_to_investigate = ['T']

    # Ground-truth agent 1
    gt_1 = [("Pr", "L"), ("Pr", "S"), ("L", "Pow"), ("S", "H"), ("H", "Pow"), ("S", "C"), ("C", "Pow")]
    # Ground-truth agent 2
    gt_2 = [("CO", "A"), ("CO2", "A"), ("A", "W"), ("B", "W"), ("O", "T"), ("W", "T")]
    # Ground-truth agent 2 post request
    gt_3 = [("H", "T"), ("C", "T"), ("CO", "A"), ("CO2", "A"), ("A", "W"), ("B", "W"), ("O", "T"), ("W", "T")]

    # Run the entire algorithm
    model_1, model_2, new_edges, elapsed_time = run(agent_1, agent_2, gt_1, gt_2, gt_3, nodes_to_investigate, timestamp)
