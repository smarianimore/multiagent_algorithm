from graphviz import Digraph, Graph

import os
#os.environ["PATH"] += os.pathsep + 'C:\\Users\\pakyr\\.conda\\envs\\bayesianEnv\\Library\\bin\\graphviz'
os.environ["PATH"] += "/usr/local/Cellar/graphviz/2.44.1/lib/graphviz"


def draw(edge, directed=True):
    # node_attr={'shape': 'circle'}
    dot = Digraph(graph_attr={'rankdir': 'LR'}, format='png') if directed else Graph()
    dot.edges(edge)
    return dot


def difference(gt, pred, stat=False):
    f = Digraph(graph_attr={'rankdir': 'LR'}, format='png')
    new_edges = [ed for ed in pred if ed not in gt]
    f.attr('edge', color='blue')
    f.edges(new_edges)

    missed_edges = [ed for ed in gt if ed not in pred]
    f.attr('edge', color='red')
    f.edges(missed_edges)

    recovered_edges = [ed for ed in pred if ed in gt]
    f.attr('edge', color='green')
    f.edges(recovered_edges)

    if stat:
        return f, new_edges, missed_edges, recovered_edges
    else:
        return f