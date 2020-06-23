import networkx as nx
from qtree.graph_model.base import wrap_general_graph_for_qtree


def generate_erdos_graph(n_nodes, probability):
    """
    Generates a random graph with n_nodes and the probability of
    edge equal probability.

    Parameters
    ----------
    n_nodes : int
          Number of nodes
    probability : float
          probability of edge
    Returns
    -------
    graph : networkx.Graph
          Random graph usable by graph_models
    """

    return wrap_general_graph_for_qtree(
        nx.generators.fast_gnp_random_graph(
            n_nodes,
            probability))


def generate_grid_graph(m, n, periodic=False):
    """
    Generates a 2d grid with possible periodic boundary
    Parameters
    ----------
    m, n: int
          Grid size
    periodic: bool, default False
          If the grid should be made periodic
    """
    return wrap_general_graph_for_qtree(
        nx.generators.grid_2d_graph(m, n, periodic=periodic))
