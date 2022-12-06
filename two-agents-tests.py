# DOC This file contains a communication example between two agents
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

now = datetime.datetime.now()
#timestamp = datetime.datetime.timestamp(now)
REPORT = 'two_agents_' + str(now)
OUTPUT_DIR = "output/reproducibility/"

online_parameters = {
    'max_cond_vars': 4,
    'do_size': 3,
    'do_conf': 0.6,
    'ci_conf': 0.1
}

agent_1red_params = {
    'max_cond_vars': 4,
    'do_size': 500,
    'do_conf': 0.9,
    'ci_conf': 0.4
}

agent_2blue_params = {
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
    gt_edges = gt  # NB was 'pred'
    pred_nodes = nodes_from_edges(pred)
    pred_edges = pred

    # Print figures
    dot, fp, fn, tp = difference(gt, pred, stat=True)
    # dot.view(directory=f'tmp/test/{output_name}')

    gt_net = f"\nGround-truth \nNodes: {gt_nodes}\tlen={len(gt_nodes)} \nEdges: {gt_edges}\tlen={len(gt_edges)}"
    pred_net = f"\nPredicted \nNodes: {pred_nodes}\tlen={len(pred_nodes)} \nEdges: {pred_edges}\tlen={len(pred_edges)}"
    missed = f"\nMissed edges: {fn}"
    comp_time = f"\nComputational time: {elapsed_time} s"
    results = gt_net + pred_net + missed + comp_time
    append_to_report(results, file=REPORT)

    # Edge statistics
    n_edges = len(gt)  # Ground-truth number of edges
    fp = len(fp)
    fn = len(fn)
    tp = len(tp)

    # Performance measure
    edge_results = f'\nFalse positives: {fp} \nFalse negatives: {fn} \nTrue positives: {tp}'
    append_to_report(edge_results, file=REPORT)
    recover_rate = f'\nRecover rate: {(tp / n_edges) * 100} %'
    append_to_report(recover_rate, file=REPORT)
    missed_rate = f'\nMissed rate: {(fn / n_edges) * 100} %'
    append_to_report(missed_rate, file=REPORT)
    # shd = f'\nSHD mod: {fp + fn + unknown} %'  # TODO
    # append_to_report(shd, file=REPORT)


# 2-agent protocol run
def run(agent_1, agent_2, gt_1, gt_2, gt_3, nodes_to_investigate):
    start = time.time()
    # 1 - Local learning (Agent 1 and Agent 2)
    model_1, undirected_edges_1 = agent_1.learning(#nodes=agent_1.nodes, non_doable=agent_1.non_doable,
                                                   parameters=agent_1red_params, mod='offline',
                                                    #bn=agent_1.gt_bn, obs_data=agent_1.obs_data
                                                    edges=None)  # NB was missing
    time_1 = (time.time() - start)  # local learning time consumption
    agent_1.replace_edges(list(model_1.edges()))
    agent_1.add_undirected_edges(undirected_edges_1)
    dot = difference(gt_1, model_1.edges())  # compare gt and learnt
    dot.view(filename=f"{REPORT}_1", directory=OUTPUT_DIR)
    write_results(gt_1, model_1.edges(), time_1, "\nAgent 1 network")

    start = time.time()
    model_2, undirected_edges_2 = agent_2.learning(#nodes=agent_2.nodes, non_doable=agent_2.non_doable,
                                                   parameters=agent_2blue_params, mod='offline',
                                                    #bn=agent_2.gt_bn, obs_data=agent_2.obs_data
                                                    edges=None)
    time_2 = (time.time() - start)  # local learning time consumption
    agent_2.replace_edges(list(model_2.edges()))
    agent_2.add_undirected_edges(undirected_edges_2)
    dot = difference(gt_2, model_2.edges())  # compare gt and learnt
    dot.view(filename=f"{REPORT}_2", directory=OUTPUT_DIR)
    write_results(gt_2, model_2.edges(), time_2, "\nAgent 2 network")

    # 2 - Request (Agent 2)
    if len(nodes_to_investigate) != 0 or len(agent_2.undirected_edges) != 0:
        # Nodes in the message are the ones indicated from the user (still manually) and the ones
        # from undirected edges discovered by the local learning algorithm
        msg = agent_2.build_request_msg(nodes_to_investigate, agent_2.undirected_edges)
        append_to_report(f'\n\nUndirected edges: {agent_2.undirected_edges}', file=f"{REPORT}.txt")
        append_to_report(f'\nNodes to investigate: {nodes_to_investigate}', file=f"{REPORT}.txt")
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
        model, undirected_edges = agent_1.learning(#nodes=new_nodes, non_doable=new_non_doable,
                                                   parameters=online_parameters, mod='online',
                                                    #bn=None, obs_data=new_data
                                                    edges=new_edges)
        time_3 = (time.time() - start)
        # The response contains the learnt edges
        response = agent_1.build_response_msg(model.edges())
        append_to_report(f'\nOnline predicted edges: {model.edges()}', file=f"{REPORT}.txt")
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
    dot.view(filename=f"{REPORT}_3", directory=OUTPUT_DIR)
    write_results(gt_3, agent_2.edges, time_3, "\nAgent 2 network after request")

    return model_1, model_2, response['edges'], elapsed_time


if __name__ == '__main__':
    append_to_report(f'\n\n#### {now}', file=f"{REPORT}.txt")
    append_to_report(f'\ntimestamp: {now}', file=f"{REPORT}.txt")
    append_to_report(f'\nMulti-agent learning', file=f"{REPORT}.txt")

    notes = ""
    append_to_report(f'\nNotes:\t{notes}', file=f"{REPORT}.txt")

    append_to_report('\n\nOffline agent 1 parameters:', file=f"{REPORT}.txt")
    for par in agent_1red_params:
        append_to_report(f'\n{par} = {agent_1red_params[par]}', file=f"{REPORT}.txt")

    append_to_report('\n\nOffline agent 2 parameters:', file=f"{REPORT}.txt")
    for par in agent_2blue_params:
        append_to_report(f'\n{par} = {agent_2blue_params[par]}', file=f"{REPORT}.txt")

    append_to_report('\n\nOnline parameters:', file=f"{REPORT}.txt")
    for par in online_parameters:
        append_to_report(f'\n{par} = {online_parameters[par]}', file=f"{REPORT}.txt")

    # Choose how to divide the network
    net_1red = networks.create_gt_net_skel(['Pr', 'L', 'Pow', 'S', 'H', 'C'], False)
    net_2blue = networks.create_gt_net_skel(['W', 'O', 'T', 'B', 'A', 'CO', 'CO2'], False)

    # Initialize agents
    agent_1red = Agent(nodes=net_1red['nodes'], non_doable=net_1red['non_doable'], gt_edges=net_1red['edges'],
                       obs_data=net_1red['dataset'])
    agent_2blue = Agent(nodes=net_2blue['nodes'], non_doable=net_2blue['non_doable'], gt_edges=net_2blue['edges'],
                        obs_data=net_2blue['dataset'])

    # List the nodes to send in the message from agent 2 to agent 1
    # So far, this array is managed manually, but the idea is to build an automatic method to recognize the nodes to
    # investigate from outliers values
    frontier = ['T']

    # Ground-truth agent 1
    gt_1red_pre = [("Pr", "L"), ("Pr", "S"), ("L", "Pow"), ("S", "H"), ("H", "Pow"), ("S", "C"), ("C", "Pow")]
    # Ground-truth agent 2
    gt_2blue = [("CO", "A"), ("CO2", "A"), ("A", "W"), ("B", "W"), ("O", "T"), ("W", "T")]
    # Ground-truth agent 2 post request
    gt_2blue_post = [("H", "T"), ("C", "T"), ("CO", "A"), ("CO2", "A"), ("A", "W"), ("B", "W"), ("O", "T"), ("W", "T")]

    # Run the entire algorithm
    model_1red, model_2blue, new_edges, elapsed_time = run(agent_1red, agent_2blue, gt_1red_pre, gt_2blue, gt_2blue_post, frontier)
