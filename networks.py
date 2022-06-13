# File containing the definitions of the various defined networks

import pandas as pd

# Ground-truth information: observational data, edges and non-doable nodes
#dataset = pd.read_csv("ocik\\demo\\store\\test\\indexed.csv", sep=',')
dataset = pd.read_csv("ocik/demo/store/test/indexed.csv", sep=',')
all_edges = [("Pr", "L"), ("Pr", "S"), ("L", "Pow"), ("S", "H"), ("H", "Pow"), ("S", "C"),
            ("C", "Pow"), ("H", "T"), ("C", "T"), ("CO", "A"), ("CO2", "A"), ("A", "W"), ("B", "W"),
            ("O", "T"), ("W", "T")]
all_non_doable = ['Pr', 'Pow', 'T', 'CO', 'CO2', 'O']
test = False  # True when we are testing the algorithm, False otherwise


# Get network from nodes
def get_network_from_nodes(nodes, test):
    edges = []
    non_doable = []

    for edge in all_edges:
        if edge[0] in nodes and edge[1] in nodes:
            edges.append(edge)

    for node in nodes:
        if node in all_non_doable:
            non_doable.append(node)

    df = dataset.drop(columns=[x for x in dataset.columns if x not in nodes])

    network = {
        "nodes": nodes,
        "edges": all_edges if test else edges,
        "non_doable": all_non_doable if test else non_doable,
        "dataset": dataset if test else df
    }

    return network


