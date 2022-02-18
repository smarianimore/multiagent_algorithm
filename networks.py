# File containing the definitions of the various defined networks

import pandas as pd

dataset = pd.read_csv("ocik\\demo\\store\\test\\network.csv", sep=',')

complete = [("Pr", "L"), ("Pr", "S"), ("L", "Pow"), ("S", "H"), ("H", "Pow"), ("S", "C"),
            ("C", "Pow"), ("H", "T"), ("C", "T"), ("CO", "A"), ("CO2", "A"), ("A", "W"), ("B", "W"),
            ("O", "T"), ("W", "T")]

test = True

nodes_a = ['Pr', 'L', 'Pow', 'H', 'C', 'S']
network_a = {
    "nodes": nodes_a,
    "edges": complete if test else [("Pr", "L"), ("Pr", "S"), ("L", "Pow"), ("S", "H"), ("H", "Pow"), ("S", "C"), ("C", "Pow")],
    "non_doable": ['Pr', 'Pow'],
    "dataset": dataset if test else dataset.drop(columns=[x for x in dataset.columns if x not in nodes_a])
}
nodes_b = ['CO', 'CO2', 'A', 'W', 'B', 'T', 'O']
network_b = {
    "nodes": nodes_b,
    "edges": complete if test else [("CO", "A"), ("CO2", "A"), ("A", "W"), ("B", "W"), ("O", "T"), ("W", "T")],
    "non_doable": ['CO', 'CO2', 'T'],
    "dataset": dataset if test else dataset.drop(columns=[x for x in dataset.columns if x not in nodes_b])
}

network_c = {
    "nodes": ['Pr', 'L', 'Pow', 'H', 'C', 'S', 'T'],
    "edges": complete,
    "non_doable": ['Pr', 'Pow'],
    "dataset": dataset
}

network_d = {
    "nodes": ['CO', 'CO2', 'A', 'W', 'B', 'T', 'O'],
    "edges": complete,
    "non_doable": ['CO', 'CO2', 'T'],
    "dataset": dataset
}

network_e = {
    "nodes": ['A', 'CO', 'CO2', 'W'],
    "edges": complete,
    "non_doable": ['W', 'A'],
    "dataset": dataset
}

network_f = {
    "nodes": ['B', 'W', 'T', 'O', 'A'],
    "edges": complete,
    "non_doable": ['T', 'W', 'A'],
    "dataset": dataset
}
