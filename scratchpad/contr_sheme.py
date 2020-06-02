import fire
import json
import networkx as nx
import numpy as np
import copy

import utils_qaoa
import expr_json
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
    order = np.roll(
        range(G.number_of_nodes())
        ,5
    )

    tensors = expr_json.get_tensors_from_graph(G)
    mapping = {v:i for i, v in enumerate(order)}

    tensors_orig = copy.deepcopy(tensors)
    #tensors are reordered in place
    edges_orig = list(G.edges.data(data='tensor', keys=True))
    W = nx.relabel_nodes(G, mapping)
    edges_new = list(W.edges.data(data='tensor', keys=True))
    tensors_new = reorder_tensors(tensors, order)
    assert tensors_new is tensors
    mapped_idx = lambda tensor: tuple([mapping[x] for x in tensor['indices']])

    for to, tn in zip(tensors_orig, tensors_new):
        assert mapped_idx(tn) != tn['indices']
        assert mapped_idx(to) == tn['indices']

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
    peo = qtree.graph_model.get_peo(graph, int_vars=True)
    tensors



def as_json(size
            , qaoa_layers=1
            , type='randomreg', degree=3, seed=42
            , repr_way='dict_of_dicts'
           ):

    """
    Return a JSON representation of a graph.

    repr_way: 'dict_of_lists' or 'dict_of_dicts'
    """
    types_allowed = ['randomreg', 'grid', 'rectgrid', 'randomgnp']
    if type in types_allowed:
        args = dict(S=size
                    , p = qaoa_layers
                    , type=type
                    , degree=degree
                    , seed=seed
                   )
        G, n_qubits = utils_qaoa.get_test_expr_graph(**args)
        #dict_ = nx.to_dict_of_dicts(G)
        if repr_way=='dict_of_dicts':
            dict_ = nx.to_dict_of_dicts(G)
        elif repr_way=='dict_of_lists':
            dict_ = nx.to_dict_of_lists(G)
        to_db =  {}
        # TODO: generator of id should be separate
        to_db['_id'] = f"p{qaoa_layers}_expr.S{size}_{type}_d{degree}_s{seed}"
        # Note: mongodb will try to use all the nested dicts,
        #       so store the graph as string
        to_db['tensors'] = get_tensors_from_graph(G, remove_data_key=True)

        to_db['graph'] = json.dumps(dict_)
        to_db['n_qubits'] = n_qubits
        to_db['extra'] = args
        to_db['tags'] = ['qaoa', 'maxCut', 'expr']
        str = json.dumps(to_db)
        print(str)

    else:
        raise Exception(f"Invalid graph type {type}, should be one of {types_allowed}")

print(__name__)
if __name__ == "__main__":
    fire.Fire(as_json)

