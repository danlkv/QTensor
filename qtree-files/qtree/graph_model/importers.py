"""
Conversion from other data structures to the graphs supported by
qtree
"""
import networkx as nx
import itertools
import re
import lzma
from io import StringIO

from .base import remove_node
from qtree.optimizer import Var
from qtree.logger_setup import log


def circ2graph(qubit_count, circuit, pdict={}, max_depth=None,
               omit_terminals=True):
    """
    Constructs a graph from a circuit in the form of a
    list of lists.

    Parameters
    ----------
    qubit_count : int
            number of qubits in the circuit
    circuit : list of lists
            quantum circuit as returned by
            :py:meth:`operators.read_circuit_file`
    pdict : dict
            Dictionary with placeholders if any parameteric gates
            were unresolved
    max_depth : int, default None
            Maximal depth of the circuit which should be used
    omit_terminals : bool, default True
            If terminal nodes should be excluded from the final
            graph.

    Returns
    -------
    graph : networkx.MultiGraph
            Graph which corresponds to the circuit
    data_dict : dict
            Dictionary with all tensor data
    """
    import functools
    import qtree.operators as ops

    if max_depth is None:
        max_depth = len(circuit)

    data_dict = {}

    # Let's build the graph.
    # The circuit is built from left to right, as it operates
    # on the ket ( |0> ) from the left. We thus first place
    # the bra ( <x| ) and then put gates in the reverse order

    # Fill the variable `frame`
    layer_variables = list(range(qubit_count))
    current_var_idx = qubit_count

    # Initialize the graph
    graph = nx.MultiGraph()

    # Populate nodes and save variables of the bra
    bra_variables = []
    for var in layer_variables:
        graph.add_node(var, name=f'o_{var}', size=2)
        bra_variables.append(Var(var, name=f"o_{var}"))

    # Place safeguard measurement circuits before and after
    # the circuit
    measurement_circ = [[ops.M(qubit) for qubit in range(qubit_count)]]

    combined_circ = functools.reduce(
        lambda x, y: itertools.chain(x, y),
        [measurement_circ, reversed(circuit[:max_depth])])

    # Start building the graph in reverse order
    for layer in combined_circ:
        for op in layer:
            # build the indices of the gate. If gate
            # changes the basis of a qubit, a new variable
            # has to be introduced and current_var_idx is increased.
            # The order of indices
            # is always (a_new, a, b_new, b, ...), as
            # this is how gate tensors are chosen to be stored
            variables = []
            current_var_idx_copy = current_var_idx
            for qubit in op.qubits:
                if qubit in op.changed_qubits:
                    variables.extend(
                        [layer_variables[qubit],
                         current_var_idx_copy])
                    graph.add_node(
                        current_var_idx_copy,
                        name='v_{}'.format(current_var_idx_copy),
                        size=2)
                    current_var_idx_copy += 1
                else:
                    variables.extend([layer_variables[qubit]])

            # Form a tensor and add a clique to the graph
            # fill placeholders in gate's parameters if any
            for par, value in op.parameters.items():
                if isinstance(value, ops.placeholder):
                    op._parameters[par] = pdict[value]

            data_key = hash((op.name, tuple(op.parameters.items())))
            tensor = {'name': op.name, 'indices': tuple(variables),
                      'data_key': data_key}

            # Insert tensor data into data dict
            data_dict[data_key] = op.gen_tensor()

            if len(variables) > 1:
                edges = itertools.combinations(variables, 2)
            else:
                edges = [(variables[0], variables[0])]

            graph.add_edges_from(edges, tensor=tensor)

            # Update current variable frame
            for qubit in op.changed_qubits:
                layer_variables[qubit] = current_var_idx
                current_var_idx += 1

    # Finally go over the qubits, append measurement gates
    # and collect ket variables
    ket_variables = []

    op = ops.M(0)  # create a single measurement gate object

    for qubit in range(qubit_count):
        var = layer_variables[qubit]
        new_var = current_var_idx

        ket_variables.append(Var(new_var, name=f'i_{qubit}', size=2))
        # update graph and variable `frame`
        graph.add_node(new_var, name=f'i_{qubit}', size=2)
        data_key = hash((op.name, tuple(op.parameters.items())))
        tensor = {'name': op.name, 'indices': (var, new_var),
                  'data_key': data_key}

        graph.add_edge(var, new_var, tensor=tensor)
        layer_variables[qubit] = new_var
        current_var_idx += 1

    if omit_terminals:
        graph.remove_nodes_from(
            tuple(map(int, bra_variables + ket_variables)))

    v = graph.number_of_nodes()
    e = graph.number_of_edges()

    log.info(f"Generated graph with {v} nodes and {e} edges")
    # log.info(f"last index contains from {layer_variables}")

    return graph, data_dict, bra_variables, ket_variables


def buckets2graph(buckets, ignore_variables=[]):
    """
    Takes buckets and produces a corresponding undirected graph. Single
    variable tensors are coded as self loops and there may be
    multiple parallel edges.

    Parameters
    ----------
    buckets : list of lists
    ignore_variables : list, optional
       Variables to be deleted from the resulting graph.
       Numbering is 0-based.

    Returns
    -------
    graph : networkx.MultiGraph
            contraction graph of the circuit
    """
    # convert everything to int
    ignore_variables = [int(var) for var in ignore_variables]

    graph = nx.MultiGraph()

    # Let's build an undirected graph for variables
    for n, bucket in enumerate(buckets):
        for tensor in bucket:
            new_nodes = []
            for idx in tensor.indices:
                # This may reintroduce the same node many times,
                # be careful if using something other than
                graph.add_node(int(idx), name=idx.name, size=idx.size)
                new_nodes.append(int(idx))
            if len(new_nodes) > 1:
                edges = itertools.combinations(new_nodes, 2)
            else:
                # If this is a single variable tensor, add self loop
                node = new_nodes[0]
                edges = [[node, node]]
            graph.add_edges_from(
                edges,
                tensor={
                    'name': tensor.name,
                    'indices': tuple(map(int, tensor.indices)),
                    'data_key': tensor.data_key
                    }
            )

    # Delete any requested variables from the final graph
    if len(ignore_variables) > 0:
        for var in ignore_variables:
            remove_node(graph, var)

    return graph


def read_gr_file(file_or_data, as_data=False, compressed=False):
    """
    Reads graph from a DGF/GR file
    The file can be specified through the filename or
    its data can be given to this function directly
    Parameters
    ----------
    file_or_data: str
             Name of the file with graph data or its contensts
    as_data: bool, default False
             If filedata should be interpreted as contents of the
             data file
    compressed : bool
           if input file or data is compressed
    """
    import sys

    ENCODING = sys.getdefaultencoding()

    graph = nx.Graph()
    if as_data is False:
        if compressed:
            datafile = lzma.open(file_or_data, 'r')
        else:
            datafile = open(file_or_data, 'r+')
    else:
        if compressed:
            datafile = StringIO(lzma.decompress(file_or_data))
        else:
            datafile = StringIO(file_or_data)

    # search for the header line
    comment_patt = '^(\s*c\s+)(?P<comment>.*)'
    header_patt = (
        '^(\s*p\s+)((?P<file_type>cnf|tw)\s+)?(?P<n_nodes>\d+)\s+(?P<n_edges>\d+)')
    if compressed:
        # if file is compressed then bytes are read. Hence we need to
        # transform patterns to byte patterns
        comment_patt = comment_patt.encode(ENCODING)
        header_patt = header_patt.encode(ENCODING)

    for n, line in enumerate(datafile):
        m = re.search(comment_patt, line)
        if m is not None:
            continue
        m = re.search(header_patt, line)
        if m is not None:
            n_nodes = int(m.group('n_nodes'))
            n_edges = int(m.group('n_edges'))
            break
        else:
            raise ValueError(f'File format error at line {n}:\n'
                             f' expected pattern: {header_patt}')

    # add nodes as not all of them may be connected
    graph.add_nodes_from(range(1, n_nodes+1))

    # search for the edges
    edge_patt = '^(\s*e\s+)?(?P<u>\d+)\s+(?P<v>\d+)'
    if compressed:
        # if file is compressed then bytes are read. Hence we need to
        # transform patterns to byte patterns
        edge_patt = edge_patt.encode(ENCODING)

    for nn, line in enumerate(datafile, n):
        m = re.search(comment_patt, line)
        if m is not None:
            continue
        m = re.search(edge_patt, line)
        if m is None:
            raise ValueError(f'File format error at line {nn}:\n'
                             f' expected pattern: {edge_patt}')
        graph.add_edge(int(m.group('u')), int(m.group('v')))

    if (graph.number_of_edges() != n_edges):
        raise ValueError('Header states:\n'
                         f' n_nodes = {n_nodes}, n_edges = {n_edges}\n'
                         'Got graph:\n'
                         f' n_nodes = {graph.number_of_nodes()},'
                         f' n_edges = {graph.number_of_edges()}\n')
    datafile.close()
    return graph


def read_td_file(file_or_data, as_data=False, compressed=False):
    """
    Reads file/data in the td format of the PACE 2017 competition
    Returns a tree decomposition: a nx.Graph with frozensets as nodes

    Parameters
    ----------
    filedata: str
             Name of the file with graph data or its contensts
    as_data: bool, default False
             If filedata should be interpreted as contents of the
             data file
    compressed: bool
             Input file or data is compressed
    """
    graph = nx.Graph()
    if as_data is False:
        if compressed:
            datafile = lzma.open(file_or_data, 'r+')
        else:
            datafile = open(file_or_data, 'r+')
    else:
        if compressed:
            datafile = StringIO(lzma.decompress(file_or_data))
        else:
            datafile = StringIO(file_or_data)

    # search for the header line
    comment_patt = re.compile('^(\s*c\s+)(?P<comment>.*)')
    header_patt = re.compile(
        '^(\s*s\s+)(?P<file_type>td)\s+(?P<n_cliques>\d+)\s+(?P<max_clique>\d+)\s+(?P<n_nodes>\d+)')
    for n, line in enumerate(datafile):
        m = re.search(comment_patt, line)
        if m is not None:
            continue
        m = re.search(header_patt, line)
        if m is not None:
            n_nodes = int(m.group('n_nodes'))
            n_cliques = int(m.group('n_cliques'))
            treewidth = int(m.group('max_clique')) - 1
            break
        else:
            raise ValueError(f'File format error at line {n}:\n'
                             f' expected pattern: {header_patt}')

    # add nodes as not all of them may be connected
    graph.add_nodes_from(range(1, n_cliques+1))

    # search for the cliques and collect them into a dictionary
    all_nodes = set()
    clique_dict = {}
    clique_patt = re.compile('^(\s*b\s+)(?P<clique_idx>\d+)(?P<clique>( \d+)*)')
    for nn, line in zip(range(n, n+n_cliques), datafile):
        m = re.search(comment_patt, line)
        if m is not None:
            continue
        m = re.search(clique_patt, line)
        if m is None:
            raise ValueError(f'File format error at line {nn}:\n'
                             f' expected pattern: {clique_patt}')
        clique = frozenset(map(int, m.group('clique').split()))
        clique_dict[int(m.group('clique_idx'))] = clique
        all_nodes = all_nodes.union(clique)

    # search for the edges between cliques
    edge_patt = re.compile('^(\s*e\s+)?(?P<u>\d+)\s+(?P<v>\d+)')
    for nnn, line in enumerate(datafile, nn):
        m = re.search(comment_patt, line)
        if m is not None:
            continue
        m = re.search(edge_patt, line)
        if m is None:
            raise ValueError(f'File format error at line {nnn}:\n'
                             f' expected pattern: {edge_patt}')
        graph.add_edge(int(m.group('u')), int(m.group('v')))

    assert(len(all_nodes) == n_nodes)

    # finally, replace clique indices by their contents
    graph = nx.relabel_nodes(graph, clique_dict)

    datafile.close()
    return graph, treewidth


def test_read_gr_files():
    from .exporters import generate_gr_file

    data = 'p tw 9 6\n1 3\n2 5\n2 6\n4 7\n4 8\n4 9\n'
    g = read_gr_file(data, as_data=True)
    g2 = read_gr_file(generate_gr_file(g), as_data=True)
    assert(nx.isomorphism.is_isomorphic(g, g2))
