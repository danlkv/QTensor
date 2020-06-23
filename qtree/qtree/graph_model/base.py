import networkx as nx
import copy
import random
import numpy as np
import itertools
from operator import mul
from functools import reduce

import matplotlib.pyplot as plt
from qtree.logger_setup import log

random.seed(0)


def relabel_graph_nodes(graph, label_dict=None, with_data=True):
    """
    Relabel graph nodes.The graph
    is relabelled (and returned) according to the label
    dictionary and an inverted dictionary is returned.
    Only integers are allowed as labels. If some other
    objects will be passed inn the label_dict, they will
    be attempted to convert to integers. If no dictionary
    will be passed then nodes will be relabeled according to
    consequtive integers starting from 0.

    In contrast to the Networkx version this one also relabels
    indices in the 'tensor' parameter of edges

    Parameters
    ----------
    graph : networkx.Graph
            graph to relabel
    label_dict : dict-like, default None
            dictionary for relabelling {old : new}
    with_data : bool, default True
            if we will check and relabel data on the edges as well
    Returns
    -------
    new_graph : networkx.Graph
            relabeled graph
    label_dict : dict
            {new : old} dictionary for inverse relabeling
    """
    # Ensure label dictionary contains integers or create one
    if label_dict is None:
        label_dict = {int(old): num for num, old in
                      enumerate(graph.nodes(data=False))}
    else:
        label_dict = {int(key): int(val)
                      for key, val in label_dict.items()}

    tensors_hash_table = {}

    # make a deep copy. We want to change all attributes without
    # interference
    new_graph = copy.deepcopy(graph)

    if with_data:
        args_to_nx = {'data': 'tensor'}
        if graph.is_multigraph():
            args_to_nx['keys'] = True

        for edgedata in graph.edges.data(**args_to_nx):
            *edge, tensor = edgedata
            if tensor is not None:
                # create new tensor only if it was not encountered
                key = hash((tensor['data_key'],
                            tensor['indices']))
                if key not in tensors_hash_table:
                    indices = tuple(label_dict[idx]
                                    for idx in tensor['indices'])
                    new_tensor = copy.deepcopy(tensor)
                    new_tensor['indices'] = indices
                    tensors_hash_table[key] = new_tensor
                else:
                    new_tensor = tensors_hash_table[key]
                new_graph.edges[
                    edge]['tensor'] = copy.deepcopy(new_tensor)

    # Then relabel nodes.
    new_graph = nx.relabel_nodes(new_graph, label_dict, copy=True)

    # invert the dictionary
    inv_label_dict = {val: key for key, val in label_dict.items()}

    return new_graph, inv_label_dict


def get_simple_graph(graph, parallel_edges=False, self_loops=False):
    """
    Simplifies graph: MultiGraphs are converted to Graphs,
    selfloops are removed
    """
    if not parallel_edges:
        # deepcopy is critical here to copy edge dictionaries
        graph = nx.Graph(copy.deepcopy(graph), copy=False)
    if not self_loops:
        graph.remove_edges_from(graph.selfloop_edges())

    return graph


def eliminate_node(graph, node, self_loops=True):
    """
    Eliminates node according to the tensor contraction rules.
    A new clique is formed, which includes all neighbors of the node.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
            Graph containing the information about the contraction
            GETS MODIFIED IN THIS FUNCTION
    node : node to contract (such that graph can be indexed by it)
    self_loops : bool
           Whether to create selfloops on the neighbors. Default True.

    Returns
    -------
    None
    """
    # Delete node itself from the list of its neighbors.
    # This eliminates possible self loop
    neighbors_wo_node = list(graph[node])
    while node in neighbors_wo_node:
        neighbors_wo_node.remove(node)

    graph.remove_node(node)

    # prune all edges containing the removed node
    edges_to_remove = []
    args_to_nx = {'data': 'tensor', 'nbunch': neighbors_wo_node,
                  'default': {'indices': []}}

    if graph.is_multigraph():
        args_to_nx['keys'] = True

    for edgedata in graph.edges.data(**args_to_nx):
        *edge, tensor = edgedata
        if node in tensor['indices']:
            edges_to_remove.append(edge)

    graph.remove_edges_from(edges_to_remove)

    # prepare new tensor
    if len(neighbors_wo_node) > 1:
        edges = itertools.combinations(neighbors_wo_node, 2)
    elif len(neighbors_wo_node) == 1 and self_loops:
        # This node had a single neighbor, add self loop to it
        edges = [[neighbors_wo_node[0], neighbors_wo_node[0]]]
    else:
        # This node had no neighbors
        edges = None

    if edges is not None:
        graph.add_edges_from(
            edges,
            tensor={
                'name': 'E{}'.format(int(node)),
                'indices': tuple(neighbors_wo_node),
                'data_key':  None
            }
        )


def remove_node(graph, node, self_loops=True):
    """
    Eliminates node if its value was fixed

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
            Graph containing the information about the contraction
            GETS MODIFIED IN THIS FUNCTION
    node : node to contract (such that graph can be indexed by it)
    self_loops : bool
           Whether to create selfloops on the neighbors. Default True.

    Returns
    -------
    None
    """
    # Delete node itself from the list of its neighbors.
    # This eliminates possible self loop
    neighbors_wo_node = list(graph[node])
    while node in neighbors_wo_node:
        neighbors_wo_node.remove(node)

    # prune all tensors containing the removed node
    args_to_nx = {'data': 'tensor', 'nbunch': neighbors_wo_node,
                  'default': {'indices': []}}
    if graph.is_multigraph():
        args_to_nx['keys'] = True

    new_selfloops = []
    for edgedata in graph.edges.data(**args_to_nx):
        *edge, tensor = edgedata
        indices = tensor['indices']
        if node in indices:
            new_indices = tuple(idx for idx in indices if idx != node)
            tensor['indices'] = new_indices
            # Invalidate data pointer as this tensor is a slice
            tensor['data_key'] = None
            if self_loops and len(new_indices) == 1:  # create a self loop
                neighbor = new_indices[0]
                new_selfloops.append((neighbor, tensor))
            else:
                graph.edges[edge]['tensor'] = tensor

    graph.remove_node(node)

    # introduce selfloops
    if self_loops:
        for v, tensor in new_selfloops:
            graph.add_edge(v, v, tensor=tensor)


def get_cost_by_node(graph, node):
    """
    Outputs the cost corresponding to the
    contraction of the node in the graph

    Parameters
    ----------
    graph : networkx.MultiGraph
               Graph containing the information about the contraction

    node : node of the graph (such that graph can be indexed by it)

    Returns
    -------
    memory : int
              Memory cost for contraction of node
    flops : int
              Flop cost for contraction of node
    """
    neighbors_with_size = {neighbor: graph.nodes[neighbor]['size']
                           for neighbor in graph[node]}

    # We have to find all unique tensors which will be contracted
    # in this bucket. They label the edges coming from
    # the current node. Application of identical tensors many times
    # can be encoded in multiple edges between the node and its neighbor.
    # We have to count the number of unique tensors.
    tensors = []
    selfloop_tensors = []

    args_to_nx = {'data': 'tensor'}

    if graph.is_multigraph():
        args_to_nx['keys'] = True

    for edgedata in graph.edges.data(**args_to_nx):
        *edge, tensor = edgedata
        u, v, *edge_key = edge
        edge_key = edge_key[0] if edge_key != [] else 0
        # the tuple (edge_key, indices, data_key) uniquely
        # identifies a tensor
        tensors.append((
            edge_key, tensor['indices'], tensor['data_key']))
        if u == v:
            selfloop_tensors.append((
                edge_key, tensor['indices'], tensor['data_key']))

    # get unique tensors
    tensors = set(tensors)

    # Now find the size of the result.
    # Ensure the node itself from the list of its neighbors.
    # This eliminates possible self loop
    neighbors_wo_node = copy.copy(neighbors_with_size)
    while node in neighbors_wo_node:
        neighbors_wo_node.pop(node)

    # memory estimation: the size of the result + all sizes of terms
    size_of_the_result = reduce(
        mul, [val for val in neighbors_wo_node.values()], 1)
    memory = size_of_the_result
    for tensor_key in tensors:
        _, indices, _ = tensor_key
        mem = reduce(
            mul, [graph.nodes[idx]['size'] for idx in indices], 1)
        memory += mem

    # Now calculate number of FLOPS
    n_unique_tensors = len(tensors)
    assert n_unique_tensors > 0
    n_multiplications = n_unique_tensors - 1

    # There are n_multiplications and 1 addition
    # repeated size_of_the_result*size_of_contracted_variable
    # times for each contraction
    flops = (size_of_the_result *
             graph.nodes[node]['size']*(1 + n_multiplications))

    return memory, flops


def get_total_size(graph):
    """
    Calculates memory to store the tensor network
    expressed in the graph model form.

    Parameters
    ----------
    graph : networkx.MultiGraph
             Graph of the network
    Returns
    -------
    memory : int
            Amount of memory
    """
    # We have to find all unique tensors which will be contracted
    # in this bucket. They label the edges coming from
    # the current node. Application of identical tensors many times
    # can be encoded in multiple edges between the node and its neighbor.
    # We have to count the number of unique tensors.
    tensors = []
    args_to_nx = {'data': 'tensor'}

    if graph.is_multigraph():
        args_to_nx['keys'] = True

    for edgedata in graph.edges.data(**args_to_nx):
        *edge, tensor = edgedata
        u, v, *edge_key = edge
        edge_key = edge_key[0] if edge_key != [] else 0
        # the tuple (edge_key, indices, data_key) uniquely
        # identifies a tensor
        tensors.append((
            edge_key, tensor['indices'], tensor['data_key']))

    # get unique tensors
    tensors = set(tensors)

    # memory estimation
    memory = 0
    for tensor_key in tensors:
        _, indices, _ = tensor_key
        mem = np.prod([graph.nodes[idx]['size'] for idx in indices])
        memory += mem

    return memory


def get_contraction_costs(old_graph, free_vars=[]):
    """
    Estimates the cost of the bucket elimination algorithm.
    The order of elimination is defined by node order (if ints are
    used as nodes then it will be the values of integers).

    Parameters
    ----------
    old_graph : networkx.Graph or networkx.MultiGraph
               Graph containing the information about the contraction
    free_vars : list, optional
               Nodes that will be skipped
    Returns
    -------
    memory : list
              Memory cost for steps of the bucket elimination algorithm
    flops : list
              Flop cost for steps of the bucket elimination algorithm
    """
    graph = copy.deepcopy(old_graph)
    nodes = sorted(graph.nodes, key=int)
    free_vars = [int(var) for var in free_vars]

    # Early return if graph is empty
    if len(nodes) == 0:
        return [1], [1]

    results = []
    for n, node in enumerate(nodes):
        if node not in free_vars:
            memory, flops = get_cost_by_node(graph, node)
            results.append((memory, flops))
            eliminate_node(graph, node)

    # Estimate cost of the last tensor product if subsets of
    # amplitudes were evaluated
    if len(free_vars) > 0:
        size_of_the_result = len(free_vars)
        tensor_orders = [
            subgraph.number_of_nodes()
            for subgraph
            in nx.components.connected_component_subgraphs(graph)]
        # memory estimation: the size of the result + all sizes of terms
        memory = 2**size_of_the_result
        for order in tensor_orders:
            memory += 2**order
        # there are number of tensors - 1 multiplications
        n_multiplications = len(tensor_orders) - 1

        # There are n_multiplications repeated size of the result
        # times
        flops = 2**size_of_the_result*n_multiplications

        results.append((memory, flops))

    return tuple(zip(*results))


def draw_graph(graph, filename=''):
    """
    Draws graph with spectral layout
    Parameters
    ----------
    graph : networkx.Graph
            graph to draw
    filename : str, default ''
            filename for image output.
            If empty string is passed the graph is displayed
    """
    fig = plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(graph)
    # pos = nx.spectral_layout(graph)

    try:
        node_color = list(map(int, graph.nodes()))
    except TypeError:
        node_color = list(range(graph.number_of_nodes()))

    nx.draw(graph, pos,
            node_color=node_color,
            node_size=100,
            cmap=plt.cm.Blues,
            with_labels=True
    )

    if len(filename) == 0:
        plt.show()
    else:
        plt.savefig(filename)


def wrap_general_graph_for_qtree(graph):
    """
    Modifies a general networkx graph to be compatible with
    graph functions from qtree. Basically, we just renumerate nodes
    from 1 and set attributes.

    Parameters
    ----------
    graph : networkx.Graph or networkx.Multigraph
            Input graph
    Returns
    -------
    new_graph : type(graph)
            Modified graph
    """
    graph_type = type(graph)

    # relabel nodes starting to integers
    label_dict = dict(zip(
        list(sorted(graph.nodes)),
        range(graph.number_of_nodes())
    ))

    # Add size to nodes
    for node in graph.nodes:
        graph.nodes[node]['size'] = 2

    # Add tensors to edges and ensure the graph is a Multigraph
    new_graph = graph_type(
        nx.relabel_nodes(graph, label_dict, copy=True)
        )

    params = {'keys': True} if graph.is_multigraph() else dict()

    for edge in new_graph.edges(**params):
        new_graph.edges[edge].update(
            {'tensor':
             {'name': 'W', 'indices': tuple(edge), 'data_key': None}})

    node_names = dict((node, f'v_{node}') for node in new_graph.nodes)
    nx.set_node_attributes(new_graph, node_names, name='name')

    return new_graph


def make_clique_on(old_graph, clique_nodes, name_prefix='C'):
    """
    Adds a clique on the specified indices. No checks is
    done whether some edges exist in the clique. The name
    of the clique is formed from the name_prefix and the
    lowest element in the clique_nodes

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
            graph to modify
    clique_nodes : list
            list of nodes to include into clique
    name_prefix : str
            prefix for the clique name
    Returns
    -------
    new_graph : type(graph)
            New graph with clique
    """
    clique_nodes = tuple(int(var) for var in clique_nodes)
    graph = copy.deepcopy(old_graph)

    if len(clique_nodes) == 0:
        return graph

    edges = [tuple(sorted(edge)) for edge in
             itertools.combinations(clique_nodes, 2)]
    node_idx = min(clique_nodes)
    graph.add_edges_from(edges,
                         tensor={'name': name_prefix + f'{node_idx}',
                                 'indices': clique_nodes,
                                 'data_key': None}
    )
    clique_size = len(clique_nodes)
    log.info(f"Clique of size {clique_size} on vertices: {clique_nodes}")

    return graph
