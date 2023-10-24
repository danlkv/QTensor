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
        type: type of graph (random, grid2d, line)

    Returns
        (nx.Graph, gamma, beta)
    """
    if type == 'random':
        G = nx.random_regular_graph(d, n)
    elif type == 'grid2d':
        G = nx.grid_2d_graph(n,n)
    elif type == 'line':
        G = nx.Graph()
        G.add_edges_from(zip(range(n-1), range(1, n)))
    else:
        raise ValueError('Unknown graph type, use random, grid2d or line')
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p
    G = nx.convert_node_labels_to_integers(G)
    return G, gamma, beta

@lru_cache
def get_test_1d_problem(N=10, d=5, Builder=None):
    if Builder is None:
        from qtensor.OpFactory import QtreeBuilder
        Builder = QtreeBuilder
    builder = Builder(N)
    for l in range(d*2):
        for i in range(N):
            builder.apply_gate(builder.operators.YPhase, i, alpha=.25)
        s = l % 2
        for i in range(N//2-s):
            u, v = s+2*i, s+2*i + 1
            builder.apply_gate(builder.operators.cX, u, v)
    return builder.circuit

