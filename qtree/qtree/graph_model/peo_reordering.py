"""
This module implements the PEO reordering procedure as described in
the article `An adaptive algorithm for quantum circuit simulation`
by R.Schutski, D.Lykov, I.Oseledets
"""

import numpy as np
import networkx as nx
import itertools
import copy
from qtree.optimizer import Var

from qtree.graph_model.base import relabel_graph_nodes, get_simple_graph
from qtree.graph_model.base import make_clique_on
from qtree.graph_model.peo_calculation import (
    get_peo, get_treewidth_from_peo)
from qtree.graph_model.importers import circ2graph


def get_fillin_graph(old_graph, peo):
    """
    Provided a graph and an order of its indices, returns a
    triangulation of that graph corresponding to the order.

    Parameters
    ----------
    old_graph : nx.Graph or nx.MultiGraph
                graph to triangulate
    peo : elimination order to use for triangulation

    Returns
    -------
    nx.Graph or nx.MultiGraph
                triangulated graph
    """
    # Ensure PEO is a list of ints
    peo = list(map(int, peo))

    # get a copy of graph in the elimination order. We do not relabel
    # tensor parameters of edges as it takes too much time
    number_of_nodes = len(peo)
    assert number_of_nodes == old_graph.number_of_nodes()
    graph, inv_label_dict = relabel_graph_nodes(
        old_graph,
        dict(zip(peo, sorted(old_graph.nodes))),
        with_data=False)

    # go over nodes and make adjacent all nodes higher in the order
    for node in sorted(graph.nodes):
        neighbors = list(graph[node])
        higher_neighbors = [neighbor for neighbor in neighbors
                            if neighbor > node]

        # form all pairs of higher neighbors
        if len(higher_neighbors) > 1:
            edges = itertools.combinations(higher_neighbors, 2)

            existing_edges = graph.edges(higher_neighbors, data=False)
            # Do not add edges over existing edges. This is
            # done to work properly with MultiGraphs
            fillin_edges = [edge for edge
                            in edges if edge not in existing_edges]
        else:
            fillin_edges = None

        # Add edges between all neighbors
        if fillin_edges is not None:
            tensor = {'name': 'C{}'.format(node),
                      'indices': (node,) + tuple(neighbors),
                      'data_key': None}
            graph.add_edges_from(
                fillin_edges, tensor=tensor
            )

    # relabel graph back so peo is a correct elimination order
    # of the resulting chordal graph
    graph, _ = relabel_graph_nodes(
        graph, inv_label_dict, with_data=False)

    return graph


def get_fillin_graph2(old_graph, peo):
    """
    Provided a graph and an order of its indices, returns a
    triangulation of that graph corresponding to the order.

    The algorithm is copied from
    "Simple Linear Time Algorithm To Test Chordality of Graph"
    by R. E. Tarjan and M. Yannakakis

    Parameters
    ----------
    old_graph : nx.Graph or nx.MultiGraph
                graph to triangulate
    peo : elimination order to use for triangulation

    Returns
    -------
    nx.Graph or nx.MultiGraph
                triangulated graph
    """
    # Ensure PEO is a list of ints
    peo = list(map(int, peo))
    peo_to_conseq = dict(zip(peo, range(len(peo))))

    number_of_nodes = len(peo)
    graph = copy.deepcopy(old_graph)

    # Safeguard check. May be removed for partial triangulation
    assert number_of_nodes == graph.number_of_nodes()

    index = [0 for ii in range(number_of_nodes)]
    f = [0 for ii in range(number_of_nodes)]

    for ii in range(number_of_nodes):
        w = peo[ii]
        idx_w = peo_to_conseq[w]
        f[idx_w] = w
        index[idx_w] = ii
        neighbors = list(graph[w])
        lower_neighbors = [v for v in neighbors
                           if peo.index(v) < ii]
        for v in lower_neighbors:
            x = v
            idx_x = peo_to_conseq[x]
            while index[idx_x] < ii:
                index[idx_x] = ii
                # Check that edge does not exist
                # Tensors added here may not correspond to cliques!
                # Their names are made incompatible with Tensorflow
                # to highlight it
                if (x, w) not in graph.edges(w):
                    tensor = {'name': 'C{}'.format(w),
                              'indices': (w, ) + tuple(neighbors),
                              'data_key': None}
                    graph.add_edge(
                        x, w,
                        tensor=tensor)
                x = f[idx_x]
                idx_x = peo_to_conseq[x]
            if f[idx_x] == x:
                f[idx_x] = w
    return graph


def is_peo_zero_fillin(old_graph, peo):
    """
    Test if the elimination order corresponds to the zero
    fillin of the graph.

    Parameters
    ----------
    graph : nx.Graph or nx.MultiGraph
                triangulated graph to test
    peo : elimination order to use for testing

    Returns
    -------
    bool
            True if elimination order has zero fillin
    """
    # Ensure PEO is a list of ints
    peo = list(map(int, peo))

    # get a copy of graph in the elimination order
    graph, label_dict = relabel_graph_nodes(
        old_graph, dict(zip(peo, sorted(old_graph.nodes())))
        )

    # go over nodes and make adjacent all nodes higher in the order
    for node in sorted(graph.nodes):
        neighbors = list(graph[node])
        higher_neighbors = [neighbor for neighbor
                            in neighbors
                            if neighbor > node]

        # form all pairs of higher neighbors
        if len(higher_neighbors) > 1:
            edges = itertools.combinations(higher_neighbors, 2)

            # Do not add edges over existing edges. This is
            # done to work properly with MultiGraphs
            existing_edges = graph.edges(higher_neighbors)
            fillin_edges = [edge for edge
                            in edges if edge not in existing_edges]
        else:
            fillin_edges = []

        # Add edges between all neighbors
        if len(fillin_edges) > 0:
            return False
    return True


def is_peo_zero_fillin2(graph, peo):
    """
    Test if the elimination order corresponds to the zero
    fillin of the graph.

    Parameters
    ----------
    graph : nx.Graph or nx.MultiGraph
                triangulated graph to test
    peo : elimination order to use for testing

    Returns
    -------
    bool
            True if elimination order has zero fillin
    """
    # Ensure PEO is a list of ints
    peo = list(map(int, peo))
    peo_to_conseq = dict(zip(peo, range(len(peo))))

    number_of_nodes = len(peo)

    index = [0 for ii in range(number_of_nodes)]
    f = [0 for ii in range(number_of_nodes)]

    for ii in range(number_of_nodes):
        w = peo[ii]
        idx_w = peo_to_conseq[w]
        f[idx_w] = w
        index[idx_w] = ii
        neighbors = list(graph[w])
        lower_neighbors = [v for v in neighbors
                           if peo.index(v) < ii]
        for v in lower_neighbors:
            idx_v = peo_to_conseq[v]
            index[idx_v] = ii
            if f[idx_v] == v:
                f[idx_v] = w
        for v in lower_neighbors:
            idx_v = peo_to_conseq[v]
            if index[f[idx_v]] < ii:
                return False
    return True


def is_clique(old_graph, vertices):
    """
    Tests if vertices induce a clique in the graph
    Multigraphs are reduced to normal graphs

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
          graph
    vertices : list
          vertices which are tested
    Returns
    -------
    bool
        True if vertices induce a clique
    """
    subgraph = old_graph.subgraph(vertices)

    # Remove selfloops so the clique is well defined
    have_edges = set(subgraph.edges()) - set(subgraph.selfloop_edges())

    # Sort all edges to be in the (low, up) order
    have_edges = set([tuple(sorted(edge, key=int))
                      for edge in have_edges])

    want_edges = set([
        tuple(sorted(edge, key=int))
        for edge in itertools.combinations(vertices, 2)
    ])
    return want_edges == have_edges


def maximum_cardinality_search(
        old_graph, last_clique_vertices=[]):
    """
    This function builds elimination order of a chordal graph
    using maximum cardinality search algorithm.
    If last_clique_vertices is
    provided the algorithm will place these indices at the end
    of the elimination list in the same order as provided.

    Parameters
    ----------
    graph : nx.Graph or nx.MultiGraph
            chordal graph to build the elimination order
    last_clique_vertices : list, default []
            list of vertices to be placed at the end of
            the elimination order
    Returns
    -------
    list
        Perfect elimination order
    """
    # convert input to int
    last_clique_vertices = [int(var) for var in last_clique_vertices]

    # Check is last_clique_vertices is a clique

    graph = copy.deepcopy(old_graph)
    n_nodes = graph.number_of_nodes()

    nodes_number_of_ord_neighbors = {node: 0 for node in graph.nodes}
    # range(0, n_nodes + 1) is important here as we need n+1 lists
    # to ensure proper indexing in the case of a clique
    nodes_by_ordered_neighbors = [[] for ii in range(0, n_nodes + 1)]
    for node in graph.nodes:
        nodes_by_ordered_neighbors[0].append(node)

    last_nonempty = 0
    peo = []

    for ii in range(n_nodes, 0, -1):
        # Take any unordered node with highest cardinality
        # or the ones in the last_clique_vertices if it was provided

        if len(last_clique_vertices) > 0:
            # Forcibly select the node from the clique
            node = last_clique_vertices.pop()
            # The following should always be possible if
            # last_clique_vertices induces a clique and I understood
            # the theorem correctly. If it raises something is wrong
            # with the algorithm/input is not a clique
            try:
                nodes_by_ordered_neighbors[last_nonempty].remove(node)
            except ValueError:
                if not is_clique(graph, last_clique_vertices):
                    raise ValueError(
                        'last_clique_vertices are not a clique')
                else:
                    raise AssertionError('Algorithmic error. Investigate')
        else:
            node = nodes_by_ordered_neighbors[last_nonempty].pop()

        peo = [node] + peo
        nodes_number_of_ord_neighbors[node] = -1

        unordered_neighbors = [
            (neighbor, nodes_number_of_ord_neighbors[neighbor])
            for neighbor in graph[node]
            if nodes_number_of_ord_neighbors[neighbor] >= 0]

        # Increase number of ordered neighbors for all adjacent
        # unordered nodes
        for neighbor, n_ordered_neighbors in unordered_neighbors:
            nodes_by_ordered_neighbors[n_ordered_neighbors].remove(
                neighbor)
            nodes_number_of_ord_neighbors[neighbor] = (
                n_ordered_neighbors + 1)
            nodes_by_ordered_neighbors[n_ordered_neighbors + 1].append(
                neighbor)

        last_nonempty += 1
        while last_nonempty >= 0:
            if len(nodes_by_ordered_neighbors[last_nonempty]) == 0:
                last_nonempty -= 1
            else:
                break

    # Create Var objects
    peo_vars = [Var(var, size=graph.nodes[var]['size'],
                    name=graph.nodes[var]['name'])
                for var in peo]
    return peo_vars


def get_equivalent_peo_naive(graph, peo, clique_vertices):
    """
    This function returns an equivalent peo with
    the clique_indices in the rest of the new order
    """
    new_peo = copy.deepcopy(peo)
    for node in clique_vertices:
        new_peo.remove(node)

    new_peo = new_peo + clique_vertices
    return new_peo


def get_equivalent_peo(old_graph, peo, clique_vertices):
    """
    This function returns an equivalent peo with
    the clique_indices in the rest of the new order
    """
    # Ensure that the graph is simple
    graph = get_simple_graph(old_graph)

    # Complete the graph
    graph_chordal = get_fillin_graph2(graph, peo)

    # MCS will produce alternative PEO with this clique at the end
    new_peo = maximum_cardinality_search(graph_chordal,
                                         list(clique_vertices))

    return new_peo


# @utils.sequential_profile_decorator(filename='fillin_graph_cprof')
def test_get_fillin_graph():
    """
    Test graph filling using the elimination order
    """
    import time
    import qtree.operators as ops
    import os
    this_dir = os.path.dirname((os.path.abspath(__file__)))
    nq, c = ops.read_circuit_file(
        this_dir +
        '/../../test_circuits/inst/cz_v2/10x10/inst_10x10_60_1.txt'
        # 'inst_2x2_7_1.txt'
    )
    g, *_ = circ2graph(nq, c, omit_terminals=False)

    peo = np.random.permutation(g.nodes)

    tim1 = time.time()
    g1 = get_fillin_graph(g, list(peo))
    tim2 = time.time()
    g2 = get_fillin_graph2(g, list(peo))
    tim3 = time.time()

    assert nx.is_isomorphic(g1, g2)
    print(tim2 - tim1, tim3 - tim2)


def test_is_zero_fillin():
    """
    Test graph filling using the elimination order
    """
    import time
    import qtree.operators as ops
    import os
    this_dir = os.path.dirname((os.path.abspath(__file__)))
    nq, c = ops.read_circuit_file(
        this_dir +
        '/../../test_circuits/inst/cz_v2/10x10/inst_10x10_60_1.txt'
    )
    g, *_ = circ2graph(nq, c, omit_terminals=False)

    g1 = get_fillin_graph(g, list(range(g.number_of_nodes())))

    tim1 = time.time()
    print(
        is_peo_zero_fillin(g1, list(range(g.number_of_nodes()))))
    tim2 = time.time()
    print(
        is_peo_zero_fillin2(g1, list(range(g.number_of_nodes()))))
    tim3 = time.time()

    print(tim2 - tim1, tim3 - tim2)


def test_maximum_cardinality_search():
    """Test maximum cardinality search algorithm"""

    # Read graph
    import qtree.operators as ops
    import os
    this_dir = os.path.dirname((os.path.abspath(__file__)))
    nq, c = ops.read_circuit_file(
        this_dir +
        '/../../inst_2x2_7_0.txt'
    )
    old_g, *_ = circ2graph(nq, c)

    # Make random clique
    vertices = list(np.random.choice(old_g.nodes, 4, replace=False))
    while is_clique(old_g, vertices):
        vertices = list(np.random.choice(old_g.nodes, 4, replace=False))

    g = make_clique_on(old_g, vertices)

    # Make graph completion
    peo, tw = get_peo(g)
    g_chordal = get_fillin_graph2(g, peo)

    # MCS will produce alternative PEO with this clique at the end
    new_peo = maximum_cardinality_search(g_chordal, list(vertices))

    # Test if new peo is correct
    assert is_peo_zero_fillin(g_chordal, peo)
    assert is_peo_zero_fillin(g_chordal, new_peo)
    new_tw = get_treewidth_from_peo(g, new_peo)
    assert tw == new_tw

    print('peo:', peo)
    print('new_peo:', new_peo)


def test_is_clique():
    """Test is_clique"""
    import qtree.operators as ops
    import os
    this_dir = os.path.dirname((os.path.abspath(__file__)))
    nq, c = ops.read_circuit_file(
        this_dir +
        '/../../inst_2x2_7_0.txt'
    )
    g, *_ = circ2graph(nq, c)

    # select some random vertices
    vertices = list(np.random.choice(g.nodes, 4, replace=False))
    while is_clique(g, vertices):
        vertices = list(np.random.choice(g.nodes, 4, replace=False))

    g_new = make_clique_on(g, vertices)

    assert is_clique(g_new, vertices)


if __name__ == '__main__':
    test_get_fillin_graph()
    test_is_clique()
    test_is_zero_fillin()
    test_maximum_cardinality_search()
