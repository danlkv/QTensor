import networkx as nx
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=2**12)
def get_test_problem(n=10, p=2, d=3, type='random'):
    """
    Get test QAOA problem

    Args:
        n: number of nodes in graph
        p: number of qaoa cycles
        d: degree of graph
        type: type of graph

    Returns
        (nx.Graph, gamma, beta)
    """
    print('Test problem: n, p, d', n, p, d)
    if type == 'random':
        G = nx.random_regular_graph(d, n)
    elif type == 'grid2d':
        G = nx.grid_2d_graph(n,n)
    elif type == 'line':
        G = nx.Graph()
        G.add_edges_from(zip(range(n-1), range(1, n)))
    G = nx.convert_node_labels_to_integers(G)
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p
    return G, gamma, beta

