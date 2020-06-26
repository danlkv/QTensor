"""
This module implements splitting of the graphs in order to reduce their
treewidth.

"""

import numpy as np
import networkx as nx
import copy

import qtree.system_defs as defs
from qtree.optimizer import Var
from qtree.logger_setup import log
from qtree.graph_model.base import (remove_node,
                                    get_contraction_costs,
                                    relabel_graph_nodes,
                                    get_simple_graph)
from qtree.graph_model.peo_calculation import (get_peo,
                                               get_upper_bound_peo,
                                               get_treewidth_from_peo)
from qtree.graph_model.clique_trees import (
    get_tree_from_peo, find_max_cliques,
    get_subtree_by_length_width, rm_element_in_tree)


def split_graph_random(old_graph, n_var_parallel=0):
    """
    Splits a graphical model with randomly chosen nodes
    to parallelize over.

    Parameters
    ----------
    old_graph : networkx.Graph
                graph to contract (after eliminating variables which
                are parallelized over)
    n_var_parallel : int
                number of variables to eliminate by parallelization

    Returns
    -------
    idx_parallel : list of Idx
          variables removed by parallelization
    graph : networkx.Graph
          new graph without parallelized variables
    """
    graph = copy.deepcopy(old_graph)

    indices = [var for var in graph.nodes(data=False)]
    idx_parallel = np.random.choice(
        indices, size=n_var_parallel, replace=False)

    idx_parallel_var = [Var(var, size=graph.nodes[var])
                        for var in idx_parallel]

    for idx in idx_parallel:
        remove_node(graph, idx)

    log.info("Removed indices by parallelization:\n{}".format(idx_parallel))
    log.info("Removed {} variables".format(len(idx_parallel)))
    peo, treewidth = get_peo(graph)

    return sorted(idx_parallel_var, key=int), graph


def get_node_by_degree(graph):
    """
    Returns a list of pairs (node : degree) for the
    provided graph.

    Parameters
    ----------
    graph : networkx.Graph without self-loops and parallel edges

    Returns
    -------
    nodes_by_degree : dict
    """
    nodes_by_degree = list((node, degree) for
                           node, degree in graph.degree())
    return nodes_by_degree


def get_node_by_betweenness(graph):
    """
    Returns a list of pairs (node : betweenness) for the
    provided graph

    Parameters
    ----------
    graph : networkx.Graph without self-loops and parallel edges

    Returns
    -------
    nodes_by_beteenness : dict
    """
    nodes_by_betweenness = list(
        nx.betweenness_centrality(
            graph,
            normalized=False, endpoints=True).items())

    return nodes_by_betweenness


def get_node_by_mem_reduction(old_graph):
    """
    Returns a list of pairs (node : reduction_in_flop_cost) for the
    provided graph. The graph is **ASSUMED** to be in the optimal
    elimination order, e.g. the nodes have to be relabelled by
    peo

    Parameters
    ----------
    graph : networkx.Graph without self-loops and parallel edges

    Returns
    -------
    nodes_by_mem_reduction : dict
    """

    graph = copy.deepcopy(old_graph)

    # Get flop cost of the bucket elimination
    initial_mem, initial_flop = get_contraction_costs(graph)

    nodes_by_mem_reduction = []
    for node in graph.nodes(data=False):
        reduced_graph = copy.deepcopy(graph)
        # Take out one node
        remove_node(reduced_graph, node)
        mem, flop = get_contraction_costs(reduced_graph)
        delta = sum(initial_mem) - sum(mem)

        nodes_by_mem_reduction.append((node, delta))

    return nodes_by_mem_reduction


def get_node_by_treewidth_reduction(graph):
    """
    Returns a list of pairs (node : reduction_in_treewidth) for the
    provided graph. The graph is **ASSUMED** to be in the optimal
    elimination order, e.g. the nodes have to be relabelled by
    peo

    Parameters
    ----------
    graph : networkx.Graph without self-loops and parallel edges

    Returns
    -------
    nodes_by_treewidth_reduction : dict
    """
    # Get flop cost of the bucket elimination
    initial_treewidth = get_treewidth_from_peo(
        graph, sorted(graph.nodes))

    nodes_by_treewidth_reduction = []
    for node in graph.nodes(data=False):
        reduced_graph = copy.deepcopy(graph)
        # Take out one node
        remove_node(reduced_graph, node)

        treewidth = get_treewidth_from_peo(
            reduced_graph, sorted(reduced_graph.nodes))
        delta = initial_treewidth - treewidth

        nodes_by_treewidth_reduction.append((node, delta))

    return nodes_by_treewidth_reduction


def split_graph_by_metric(
        old_graph, n_var_parallel=0,
        metric_fn=get_node_by_degree,
        forbidden_nodes=()):
    """
    Parallel-splitted version of :py:meth:`get_peo` with nodes
    to split chosen according to the metric function. Metric
    function should take a graph and return a list of pairs
    (node : metric_value)

    Parameters
    ----------
    old_graph : networkx.Graph or networkx.MultiGraph
                graph to split by parallelizing over variables
                and to contract

                Parallel edges and self-loops in the graph are
                removed (if any) before the calculation of metric

    n_var_parallel : int
                number of variables to eliminate by parallelization
    metric_fn : function, optional
                function to evaluate node metric.
                Default get_node_by_degree
    forbidden_nodes : iterable, optional
                nodes in this list will not be considered
                for deletion. Default ().
    Returns
    -------
    idx_parallel : list
          variables removed by parallelization
    graph : networkx.Graph
          new graph without parallelized variables
    """
    # graph = get_simple_graph(old_graph)
    # import pdb
    # pdb.set_trace()
    graph = copy.deepcopy(old_graph)

    # convert everything to int
    forbidden_nodes = [int(var) for var in forbidden_nodes]

    # get nodes by metric in descending order
    nodes_by_metric = metric_fn(graph)
    nodes_by_metric.sort(key=lambda pair: int(pair[1]), reverse=True)

    nodes_by_metric_allowed = []
    for node, metric in nodes_by_metric:
        if node not in forbidden_nodes:
            nodes_by_metric_allowed.append((node, metric))

    idx_parallel = []
    for ii in range(n_var_parallel):
        node, metric = nodes_by_metric_allowed[ii]
        idx_parallel.append(node)

    # create var objects from nodes
    idx_parallel_var = [Var(var, size=graph.nodes[var]['size'])
                        for var in idx_parallel]

    for idx in idx_parallel:
        remove_node(graph, idx)

    log.info("Removed indices by parallelization:\n{}".format(idx_parallel))
    log.info("Removed {} variables".format(len(idx_parallel)))

    return idx_parallel_var, graph


def split_graph_with_mem_constraint_greedy(
        old_graph,
        n_var_parallel_min=0,
        mem_constraint=defs.MAXIMAL_MEMORY,
        step_by=5,
        n_var_parallel_max=None,
        metric_fn=get_node_by_mem_reduction,
        forbidden_nodes=(),
        peo_function=get_peo):
    """
    This function splits graph by greedily selecting next nodes
    up to the n_var_parallel
    using the metric function and recomputing PEO after
    each node elimination. The graph is **ASSUMED** to be in
    the perfect elimination order

    Parameters
    ----------
    old_graph : networkx.Graph()
           initial contraction graph
    n_var_parallel_min : int
           minimal number of variables to split the task to
    mem_constraint : int
           Upper limit on memory per task
    metric_function : function, optional
           function to rank nodes for elimination
    step_by : int, optional
           scan the metric function with this step
    n_var_parallel_max : int, optional
           constraint on the maximal number of parallelized
           variables. Default None
    forbidden_nodes: iterable, default ()
           nodes forbidden for parallelization
    peo_function: function
           function to calculate PEO. Should have signature
           lambda (graph): return peo, treewidth
    Returns
    -------
    idx_parallel : list
             list of removed variables
    graph : networkx.Graph
             reduced contraction graph
    """
    # convert everything to int
    forbidden_nodes = [int(var) for var in forbidden_nodes]

    graph = copy.deepcopy(old_graph)
    n_var_total = old_graph.number_of_nodes()
    if n_var_parallel_max is None:
        n_var_parallel_max = n_var_total

    mem_cost, flop_cost = get_contraction_costs(graph)
    max_mem = sum(mem_cost)

    idx_parallel = []
    idx_parallel_var = []

    steps = list(range(0, n_var_parallel_max, step_by))
    if len(steps) == 0 or (n_var_parallel_max % step_by != 0):
        steps.append(n_var_parallel_max)

    steps = [step_by] * (n_var_parallel_max // step_by)
    # append last batch to steps
    steps.append(n_var_parallel_max
                 - (n_var_parallel_max // step_by) * step_by)

    for n_parallel in steps:
        # Get optimal order
        peo, tw = peo_function(graph)
        graph_optimal, inverse_order = relabel_graph_nodes(
            graph, dict(zip(peo, range(len(peo)))))

        # get nodes by metric in descending order
        nodes_by_metric_optimal = metric_fn(graph_optimal)
        nodes_by_metric_optimal.sort(
            key=lambda pair: pair[1], reverse=True)

        nodes_by_metric_allowed = []
        for node, metric in nodes_by_metric_optimal:
            if inverse_order[node] not in forbidden_nodes:
                nodes_by_metric_allowed.append(
                    (inverse_order[node], metric))

        # Take first nodes by cost and map them back to original
        # order
        nodes_with_cost = nodes_by_metric_allowed[:n_parallel]
        if len(nodes_with_cost) > 0:
            nodes, costs = zip(*nodes_with_cost)
        else:
            nodes = []

        # Update list and update graph
        idx_parallel += nodes

        # create var objects from nodes
        idx_parallel_var += [Var(var, size=graph.nodes[var]['size'])
                             for var in nodes]

        for node in nodes:
            remove_node(graph, node)

        # Renumerate graph nodes to be consequtive ints (may be redundant)
        label_dict = dict(zip(sorted(graph.nodes),
                              range(len(graph.nodes()))))

        graph_relabelled, _ = relabel_graph_nodes(graph, label_dict)
        mem_cost, flop_cost = get_contraction_costs(graph_relabelled)

        max_mem = sum(mem_cost)

        if (max_mem <= mem_constraint
           and len(idx_parallel) >= n_var_parallel_min):
            break

    if max_mem > mem_constraint:
        raise ValueError('Maximal memory constraint is not met')

    return idx_parallel_var, graph


def split_graph_by_metric_greedy(
        old_graph, n_var_parallel=0,
        metric_fn=get_node_by_treewidth_reduction,
        greedy_step_by=1, forbidden_nodes=(), peo_function=get_peo):
    """
    This function splits graph by greedily selecting next nodes
    up to the n_var_parallel
    using the metric function and recomputing PEO after
    each node elimination

    Parameters
    ----------
    old_graph : networkx.Graph or networkx.MultiGraph
                graph to split by parallelizing over variables
                and to contract

                Parallel edges and self-loops in the graph are
                removed (if any) before the calculation of metric

    n_var_parallel : int
                number of variables to eliminate by parallelization
    metric_fn : function, optional
                function to evaluate node metric.
                Default get_node_by_mem_reduction
    greedy_step_by : int, default 1
                Step size for the greedy algorithm

    forbidden_nodes : iterable, optional
                nodes in this list will not be considered
                for deletion. Default ().
    peo_function: function
           function to calculate PEO. Should have signature
           lambda (graph): return peo, treewidth

    Returns
    -------
    idx_parallel : list
          variables removed by parallelization
    graph : networkx.Graph
          new graph without parallelized variables
    """
    # import pdb
    # pdb.set_trace()

    # convert everything to int
    forbidden_nodes = [int(var) for var in forbidden_nodes]

    # Simplify graph
    graph = get_simple_graph(old_graph)

    idx_parallel = []
    idx_parallel_var = []

    steps = [greedy_step_by] * (n_var_parallel // greedy_step_by)
    # append last batch to steps
    steps.append(n_var_parallel
                 - (n_var_parallel // greedy_step_by) * greedy_step_by)

    for n_parallel in steps:
        # Get optimal order - recalculate treewidth
        peo, tw = peo_function(graph)
        graph_optimal, inverse_order = relabel_graph_nodes(
            graph, dict(zip(peo, sorted(graph.nodes))))

        # get nodes by metric in descending order
        nodes_by_metric_optimal = metric_fn(graph_optimal)
        nodes_by_metric_optimal.sort(
            key=lambda pair: pair[1], reverse=True)

        # filter out forbidden nodes and get nodes in original order
        nodes_by_metric_allowed = []
        for node, metric in nodes_by_metric_optimal:
            if inverse_order[node] not in forbidden_nodes:
                nodes_by_metric_allowed.append(
                    (inverse_order[node], metric))

        # Take first nodes by cost and map them back to original
        # order
        nodes_with_cost = nodes_by_metric_allowed[:n_parallel]
        if len(nodes_with_cost) > 0:
            nodes, costs = zip(*nodes_with_cost)
        else:
            nodes = []

        # Update list and update graph
        idx_parallel += nodes
        # create var objects from nodes
        idx_parallel_var += [Var(var, size=graph.nodes[var]['size'])
                             for var in nodes]
        for node in nodes:
            remove_node(graph, node)

    return idx_parallel_var, graph


def split_graph_by_tree_trimming(
        old_graph, n_var_parallel):
    """
    Splits graph by removing variables from its tree decomposition.
    The graph is **ASSUMED** to be in the perfect elimination order,
    e.g. it has to be relabelled before calling split function.

    Parameters
    ----------
    old_graph : networkx.Graph or networkx.MultiGraph
                graph to split by parallelizing over variables

                Parallel edges and self-loops in the graph are
                removed (if any)

    n_var_parallel : int
                number of variables to eliminate by parallelization
    Returns
    -------
    idx_parallel : list
          variables removed by parallelization
    graph : networkx.Graph
          new graph without parallelized variables
    """
    graph = copy.deepcopy(old_graph)

    # Produce a tree from the ordering of the graph
    # and get a maximal clique
    tree = get_tree_from_peo(
        old_graph, list(range(graph.number_of_nodes())))

    cliques = list(tree.nodes())
    all_nodes = []
    for clique in cliques:
        all_nodes = clique.union(all_nodes)
    # are_subtrees_connected(tree, list(all_nodes))

    eliminated_nodes = []
    for ii in range(n_var_parallel):
        # if ii == 179:
        #     import pdb
        #     pdb.set_trace()
        max_cliques = find_max_cliques(tree, len(all_nodes))
        nodes_in_max_cliqes = [node for clique
                               in max_cliques for node in clique]
        nodes_by_subwidth = get_subtree_by_length_width(
            tree, nodes_in_max_cliqes)

        # get (node, path length, total subtree width)
        nodes_in_rmorder = [(node, len(nodes_by_subwidth[node]),
                             sum(nodes_by_subwidth[node]))
                            for node in nodes_by_subwidth]
        # sort by path length, then by total width of subtree
        nodes_in_rmorder = sorted(
            nodes_in_rmorder,
            key=lambda x: (x[1], x[2]))
        rmnode = nodes_in_rmorder[-1][0]
        tree = rm_element_in_tree(tree, rmnode)
        eliminated_nodes.append(rmnode)

        cliques = list(tree.nodes())
        all_nodes = []
        for clique in cliques:
            all_nodes = clique.union(all_nodes)
        # are_subtrees_connected(tree, list(all_nodes))

    # we are done with finding the set for removal.
    # Remove nodes from the graph and return

    graph.remove_nodes_from(eliminated_nodes)

    return eliminated_nodes, graph
