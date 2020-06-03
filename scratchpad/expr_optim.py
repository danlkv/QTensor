import networkx as nx
import numpy as np
import copy
import sys
sys.path.append('..')

import utils_qaoa
import expr
import qtree

def reorder_tensors(tensors, permutation):
    ## independent
    def permutation_to_dict(permutation):
        perm_dict = {}
        for n, idx in enumerate(permutation):
            if isinstance(idx, int) or isinstance(idx, np.int64):
                # handle int
                perm_dict[idx] = n
            else:
                # handle qtree.optimizer.Var
                if idx.name.startswith('v'):
                    perm_dict[idx] = idx.copy(n)
                else:
                    perm_dict[idx] = idx.copy(n, name=idx.name)
        return perm_dict
    ##
    perm_dict = permutation_to_dict(permutation)
    #tensors = set(tensors)

    for tensor in tensors:
        new_indices = [perm_dict[idx] for idx in tensor['indices']]
        tensor['indices'] = tuple(new_indices)
    # the dicts are mutably modified 
    return tensors


def test_reorder_tensors_in_graph():
    G, n_qubits = utils_qaoa.get_test_expr_graph(6, 1)
    # Set up an ordering without a fixed point
    order = np.roll(
        range(G.number_of_nodes())
        ,5
    )

    tensors = expr.get_tensors_from_graph(G)
    mapping = {v:i for i, v in enumerate(order)}

    W = nx.relabel_nodes(G, mapping)
    edges_orig = list(G.edges.data(data='tensor', keys=True))
    edges_new = list(W.edges.data(data='tensor', keys=True))
    # Store original tensors for testing
    tensors_orig = copy.deepcopy(tensors)
    # Tensors are reordered in place, will change both W and G tensors
    tensors_new = reorder_tensors(tensors, order)
    assert tensors_new is tensors

    mapped_idx = lambda tensor: tuple([mapping[x] for x in tensor['indices']])

    # Test if reordered tensors
    for to, tn in zip(tensors_orig, tensors_new):
        assert mapped_idx(tn) != tn['indices']
        assert mapped_idx(to) == tn['indices']

    # Test if reordered Graph
    assert edges_new[0] != edges_orig[0]
    for en, eo in zip(edges_new, edges_orig):
        assert en[0] != eo[0]
        assert en[0] == mapping[eo[0]]
        assert en[1] == mapping[eo[1]]
        assert eo[3]['indices'] == en[3]['indices']

        assert en[0] in en[3]['indices']
        assert en[1] in en[3]['indices']




def test_reorder_tensors_small():
    tensors =[
        {'indices':(1,2), 'name':'A'}
        ,{'indices':(2,2)}
        ,{'indices':(4,1,3)}
    ]

    permutation = (2,4,1,3)
    perm_tensors = reorder_tensors(tensors, permutation)
    assert perm_tensors[0]['indices'] == (2,0)
    assert perm_tensors[0]['name'] == 'A'
    assert perm_tensors[1]['indices'] == (0,0)
    assert perm_tensors[2]['indices'] == (1,2,3)

def optimize_order(graph):
    peo, tw = qtree.graph_model.get_peo(graph, int_vars=True)
    mapping = {v:i for i, v in enumerate(peo)}
    tensors = expr.get_tensors_from_graph(graph)
    reorder_tensors(tensors, peo)
    graph = nx.relabel_nodes(graph, mapping, copy=True)
    return peo, tw, graph
