from graphviz import Digraph, Graph

import os

from networkx.classes.reportviews import OutEdgeView

# TODO put in config file
os.environ["PATH"] += "/usr/local/Cellar/graphviz/2.44.1/lib/graphviz"


def draw(edge, directed=True):
    # node_attr={'shape': 'circle'}
    dot = Digraph(graph_attr={'rankdir': 'LR'}, format='png') if directed else Graph()
    dot.edges(edge)
    return dot


# TODO change 'pred' to more simple datatype, such as list[tuple[]]
def difference(gt: list[tuple[str, str]], pred: OutEdgeView, stat: bool = False) -> object:
    """

    Parameters
    ----------
    gt :
        the list of edges in the ground truth
    pred :
        the list of edges in the learnt model
    stat :
        whether accuracy statistics are to be provided

    Returns
    -------
    object
        either the difference Digraph alone, or the Digraph and a list of spurious, missed, and recovered edges

    """
    f = Digraph(graph_attr={'rankdir': 'LR'}, format='png')
    spurious_edges = [ed for ed in pred if ed not in gt]
    f.attr('edge', color='blue')  # TODO put colors in config file
    f.edges(spurious_edges)

    missed_edges = [ed for ed in gt if ed not in pred]
    f.attr('edge', color='red')
    f.edges(missed_edges)

    recovered_edges = [ed for ed in pred if ed in gt]
    f.attr('edge', color='green')
    f.edges(recovered_edges)

    if stat:
        return f, spurious_edges, missed_edges, recovered_edges
    else:
        return f
