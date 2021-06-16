from qtensor.tools.lazy_import import cvxgraphalgs as cvx
import numpy as np

import networkx as nx


def gw_solution(G: nx.Graph):
    solnGW = cvx.algorithms.goemans_williamson_weighted(G)  # Goemans Williamson algorithm
    N = G.number_of_nodes()

    solnGW_binary = np.ones(N)
    solnGW_binary[list(solnGW.left)] = -1
    return solnGW_binary


def gw_cost(G: nx.Graph):
    solnGW_binary = gw_solution(G)
    G_adj = nx.linalg.adjacency_matrix(G, nodelist=range(len(G))).toarray()
    valGW = (len(G.edges) - np.dot(solnGW_binary,G_adj.dot(solnGW_binary))/2)/2
    return valGW
