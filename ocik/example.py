from ocik.network import BayesianNetwork
from pgmpy.models import BayesianNetwork as bnet
import pandas as pd
import numpy as np


class BigRoom:
    # Used to test the algorithm with a huge network made by lots of nodes
    def __init__(self, n_nodes, edge_prob=0.2, n_states=2):
        # Build a random bayesian network with random CPD
        bayesian = bnet.get_random(n_nodes=n_nodes, edge_prob=edge_prob, n_states=n_states)

        # Conversion from Bayesian Network to ocik.network model
        self.bn = BayesianNetwork(bayesian.edges)

        # Set the CPD: so far the setting is random, but we could find a way to extract and use the CPD of the random network
        # QUESTION: the CPD is not formally correct? (e.g. probabilities do not sum up?)
        for node in self.bn.nodes():
            parents = []
            for edge in self.bn.edges():
                if node == edge[1]:
                    parents.append(edge[0])
            # print(f'Parents for node {node}: \n{parents}')

            combinations = 2 ** len(parents)
            arr = np.random.rand(2, combinations)
            for i in range(combinations):
                arr[0][i] = round(arr[0][i], 1)
                arr[1][i] = round(1 - arr[0][i], 1)
            self.bn.set_cpd(node, arr, parents)

    def get_network(self):
        return self.bn


class Test:
    __data = "ocik/demo/store/test/network_4.csv"

    def __init__(self):
        ed = lambda a, b: [f'{a}', f'{b}']

        self.bn = BayesianNetwork([ed("Pr", "L"), ed("Pr", "S"),ed("S", "H"),
                                   ed("S", "C"), ed("H", "T"),
                                   ed("C", "T"), ed("CO", "A"),
                                   ed("CO2", "A"), ed("A", "W"), ed("B", "W"),
                                   ed("O", "T"), ed("W", "T")])

        # Conditional Probability Distribution
        self.bn.set_cpd("Pr", [[0.49], [0.51]], [])
        self.bn.set_cpd("L", [[0.01, 0.99],
                              [0.99, 0.01]], ["Pr"])
        self.bn.set_cpd("S", [[0.51, 0.49],
                              [0.49, 0.51]], ["Pr"])
        self.bn.set_cpd("H", [[0.1, 0.4],
                              [0.9, 0.6]], ["S"])
        self.bn.set_cpd("C", [[0.4, 0.1],
                              [0.6, 0.9]], ["S"])
        self.bn.set_cpd("CO", [[0.6], [0.4]], [])
        self.bn.set_cpd("CO2", [[0.3], [0.7]], [])
        self.bn.set_cpd("A", [[0.1, 0.1, 0.1, 0.9],
                              [0.9, 0.9, 0.9, 0.1]], ["CO", "CO2"])
        self.bn.set_cpd("B", [[0.5], [0.5]], [])
        self.bn.set_cpd("O", [[0.5], [0.5]], [])
        self.bn.set_cpd("W", [[0.1, 0.9, 0.9, 0.9],
                              [0.9, 0.1, 0.1, 0.1]], ["A", "B"])
        self.bn.set_cpd("T", [[0.2, 0.2, 0.2, 0.3, 0.51, 0.3, 0.51, 0.5, 0.4, 0.3, 0.3, 0.2, 0.51, 0.51, 0.51, 0.51],
                              [0.8, 0.8, 0.8, 0.7, 0.49, 0.7, 0.49, 0.5, 0.6, 0.7, 0.7, 0.8, 0.49, 0.49, 0.49, 0.49]], ["O", "W", "H", "C"])

    def get_network(self):
        return self.bn

    def load_data(self, nrows=100):
        return pd.read_csv(self.__data, nrows=nrows)


class RoomComplete:
    __data = "ocik/demo/store/test/network_4.csv"

    def __init__(self):
        ed = lambda a, b: [f'{a}', f'{b}']

        self.bn = BayesianNetwork([ed("Pr", "L"), ed("Pr", "S"),
                                   ed("L", "Pow"), ed("S", "H"),
                                   ed("H", "Pow"), ed("S", "C"),
                                   ed("C", "Pow"), ed("H", "T"),
                                   ed("C", "T"), ed("CO", "A"),
                                   ed("CO2", "A"), ed("A", "W"), ed("B", "W"),
                                   ed("O", "T"), ed("W", "T")])

        # Conditional Probability Distribution
        self.bn.set_cpd("Pr", [[0.49], [0.51]], [])
        self.bn.set_cpd("L", [[0.01, 0.99],
                              [0.99, 0.01]], ["Pr"])
        self.bn.set_cpd("S", [[0.51, 0.49],
                              [0.49, 0.51]], ["Pr"])
        self.bn.set_cpd("Pow", [[0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.5, 0.9],
                                [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1]], ["L", "H", "C"])
        self.bn.set_cpd("H", [[0.1, 0.4],
                              [0.9, 0.6]], ["S"])
        self.bn.set_cpd("C", [[0.4, 0.1],
                              [0.6, 0.9]], ["S"])
        self.bn.set_cpd("CO", [[0.6], [0.4]], [])
        self.bn.set_cpd("CO2", [[0.3], [0.7]], [])
        self.bn.set_cpd("A", [[0.1, 0.1, 0.1, 0.9],
                              [0.9, 0.9, 0.9, 0.1]], ["CO", "CO2"])
        self.bn.set_cpd("B", [[0.5], [0.5]], [])
        self.bn.set_cpd("O", [[0.5], [0.5]], [])
        self.bn.set_cpd("W", [[0.1, 0.9, 0.9, 0.9],
                              [0.9, 0.1, 0.1, 0.1]], ["A", "B"])
        self.bn.set_cpd("T", [[0.2, 0.2, 0.2, 0.3, 0.51, 0.3, 0.51, 0.5, 0.4, 0.3, 0.3, 0.2, 0.51, 0.51, 0.51, 0.51],
                              [0.8, 0.8, 0.8, 0.7, 0.49, 0.7, 0.49, 0.5, 0.6, 0.7, 0.7, 0.8, 0.49, 0.49, 0.49, 0.49]], ["O", "W", "H", "C"])

    def get_network(self):
        return self.bn

    def load_data(self, nrows=100):
        return pd.read_csv(self.__data, nrows=nrows)


class RoomMiddle2:
    __data = "ocik/demo/store/test/network_3.csv"

    def __init__(self):
        ed = lambda a, b: [f'{a}', f'{b}']

        self.bn = BayesianNetwork([ed("Pr", "L"), ed("Pr", "S"),
                                   ed("L", "Pow"), ed("S", "H"),
                                   ed("H", "Pow"), ed("S", "C"),
                                   ed("C", "Pow"), ed("H", "T"),
                                   ed("C", "T"), ed("B", "W"),
                                   ed("O", "T"), ed("W", "T")])

        # Conditional Probability Distribution
        self.bn.set_cpd("Pr", [[0.49], [0.51]], [])
        self.bn.set_cpd("L", [[0.01, 0.99],
                              [0.99, 0.01]], ["Pr"])
        self.bn.set_cpd("S", [[0.51, 0.49],
                              [0.49, 0.51]], ["Pr"])
        self.bn.set_cpd("Pow", [[0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5],
                                [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5]], ["L", "H", "C"])
        self.bn.set_cpd("H", [[0.1, 0.4],
                              [0.9, 0.6]], ["S"])
        self.bn.set_cpd("C", [[0.4, 0.1],
                              [0.6, 0.9]], ["S"])
        self.bn.set_cpd("B", [[0.5], [0.5]], [])
        self.bn.set_cpd("O", [[0.5], [0.5]], [])
        self.bn.set_cpd("W", [[0.9, 0.9],
                              [0.1, 0.1]], ["B"])
        self.bn.set_cpd("T", [[0.2, 0.2, 0.2, 0.3, 0.51, 0.3, 0.51, 0.5, 0.4, 0.3, 0.3, 0.2, 0.51, 0.51, 0.51, 0.51],
                              [0.8, 0.8, 0.8, 0.7, 0.49, 0.7, 0.49, 0.5, 0.6, 0.7, 0.7, 0.8, 0.49, 0.49, 0.49, 0.49]], ["O", "W", "H", "C"])

    def get_network(self):
        return self.bn

    def load_data(self, nrows=100):
        return pd.read_csv(self.__data, nrows=nrows)


class RoomMiddle1:
    __data = "ocik/demo/store/test/network_2.csv"

    def __init__(self):
        ed = lambda a, b: [f'{a}', f'{b}']

        self.bn = BayesianNetwork([ed("Pr", "L"), ed("L", "Pow"),
                                   ed("H", "Pow"), ed("C", "Pow"),
                                   ed("H", "T"), ed("C", "T"),
                                   ed("B", "W"), ed("O", "T"), ed("W", "T")])

        # Conditional Probability Distribution
        self.bn.set_cpd("Pr", [[0.49], [0.51]], [])
        self.bn.set_cpd("L", [[0.01, 0.99],
                              [0.99, 0.01]], ["Pr"])
        self.bn.set_cpd("Pow", [[0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5],
                                [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5]], ["L", "H", "C"])
        self.bn.set_cpd("H", [[0.3], [0.7]], [])
        self.bn.set_cpd("C", [[0.3], [0.7]], [])
        self.bn.set_cpd("B", [[0.5], [0.5]], [])
        self.bn.set_cpd("O", [[0.5], [0.5]], [])
        self.bn.set_cpd("W", [[0.2, 0.9],
                              [0.8, 0.1]], ["B"])
        self.bn.set_cpd("T", [[0.2, 0.2, 0.2, 0.3, 0.51, 0.3, 0.51, 0.5, 0.4, 0.3, 0.3, 0.2, 0.51, 0.51, 0.51, 0.51],
                              [0.8, 0.8, 0.8, 0.7, 0.49, 0.7, 0.49, 0.5, 0.6, 0.7, 0.7, 0.8, 0.49, 0.49, 0.49, 0.49]], ["O", "W", "H", "C"])

    def get_network(self):
        return self.bn

    def load_data(self, nrows=100):
        return pd.read_csv(self.__data, nrows=nrows)


class RoomBase:
    __data = "ocik/demo/store/test/network_1.csv"

    def __init__(self):
        ed = lambda a, b: [f'{a}', f'{b}']

        self.bn = BayesianNetwork([ed("Pr", "L"), ed("L", "Pow"),
                                   ed("H", "Pow"), ed("H", "T"),
                                   ed("O", "T"), ed("W", "T")])

        # Conditional Probability Distribution
        self.bn.set_cpd("Pr", [[0.5], [0.5]], [])
        self.bn.set_cpd("L", [[0.08, 0.90],
                              [0.92, 0.1]], ["Pr"])
        self.bn.set_cpd("Pow", [[0.7, 0.8, 0.9, 0.9],
                                [0.3, 0.2, 0.1, 0.1]], ["L", "H"])
        self.bn.set_cpd("H", [[0.5], [0.5]], [])
        self.bn.set_cpd("O", [[0.5], [0.5]], [])
        self.bn.set_cpd("W", [[0.5], [0.5]], [])
        self.bn.set_cpd("T", [[0.09, 0.09, 0.08, 0.4, 0.9, 0.08, 0.92, 0.5],
                              [0.91, 0.91, 0.92, 0.6, 0.1, 0.92, 0.08, 0.5]], ["O", "W", "H"])

    def get_network(self):
        return self.bn

    def load_data(self, nrows=100):
        return pd.read_csv(self.__data, nrows=nrows)


class Room:
    __data = "ocik/demo/store/room.csv"

    def __init__(self):
        ed = lambda a, b: [f'{a}', f'{b}']

        bn = BayesianNetwork([("P", "H"), ("P", "L"),
                              ("H", "T"), ("W", "T")])

        bn.set_cpd("P", [[0.6], [0.4]], [])
        bn.set_cpd("W", [[0.3], [0.7]], [])

        bn.set_cpd("H", [[0.1, 0.5],
                         [0.9, 0.5]], ["P"])

        bn.set_cpd("L", [[0.2, 0.9],
                         [0.8, 0.1]], ["P"])

        bn.set_cpd("T", [[0.05, 0.3, 0.4, 0.9],
                         [0.95, 0.7, 0.6, 0.1]], ["H", "W"])
        # "T", [[... ... ...] <- T=1
        #    ,  [... ... ...]] <- T=0

        self.bn = bn

    def get_network(self):
        return self.bn

    def load_data(self, nrows=100):
        return pd.read_csv(self.__data, nrows=nrows)


class Asia:
    __data = "ocik/demo/store/asia.csv"

    def __init__(self):
        ed = lambda a, b: [f'{a}', f'{b}']

        self.bn = BayesianNetwork([ed("A", "T"), ed("T", "O"),
                                   ed("S", "L"), ed("L", "O"),
                                   ed("S", "B"), ed("B", "D"),
                                   ed("O", "X"), ed("O", "D")])

        self.bn.set_cpd("A", [[0.99], [0.01]], [])
        self.bn.set_cpd("T", [[0.99, 0.95],
                              [0.01, 0.05]], ["A"])
        self.bn.set_cpd("S", [[0.5], [0.5]], [])
        self.bn.set_cpd("L", [[0.99, 0.90],
                              [0.01, 0.10]], ["S"])
        self.bn.set_cpd("B", [[0.7, 0.4],
                              [0.3, 0.6]], ["S"])
        self.bn.set_cpd("X", [[0.95, 0.02],
                              [0.05, 0.98]], ["O"])
        self.bn.set_cpd("O", [[0.1, 0.0, 0.0, 0.0],
                              [0.9, 1.0, 1.0, 1.0]], ["T", "L"])
        self.bn.set_cpd("D", [[0.9, 0.2, 0.3, 0.1],
                              [0.1, 0.8, 0.7, 0.9]], ["O", "B"])

    def get_network(self):
        return self.bn

    def load_data(self, nrows=100):
        return pd.read_csv(self.__data, nrows=nrows)


class Circuit:
    def __init__(self):
        ed = lambda a, b: [f'x{a}', f'x{b}']

        bn = BayesianNetwork([ed(1, 4), ed(4, 8), ed(8, 9), ed(9, 11),
                              ed(2, 5), ed(2, 6), ed(6, 8), ed(6, 10), ed(10, 11),
                              ed(3, 5), ed(5, 6), ed(5, 7), ed(7, 10)])

        from ocik.utils import f_and, f_nand, f_nor, f_not, f_or, f_xor

        def fill(x, gate):
            val = int(x.name.split("=")[1])
            if len(x.index) == 1:
                return [gate(x)[val]]
            if type(x.index[0]) == tuple:
                return [int(gate(*idx) == val) for idx in x.index]
            else:
                return [int(gate(idx) == val) for idx in x.index]

        rand = lambda x: lambda u: [x, 1 - x]

        circuit = [('x1', rand(0.5)), ('x2', rand(0.5)), ('x3', rand(0.5)), ('x4', f_not), ('x5', f_nand),
                   ('x6', f_and), ('x7', f_not), ('x8', f_xor), ('x9', f_not), ('x10', f_xor), ('x11', f_xor)]

        for node, foo in circuit:
            cpd = bn.get_cpd(node)[1].apply(lambda x: fill(x, foo))
            bn._set_cpd(node, cpd)

        self.bn = bn

    def get_network(self):
        return self.bn