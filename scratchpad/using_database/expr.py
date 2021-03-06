import fire
import sys
sys.path.append('..')
import json
import networkx as nx

import utils_qaoa

def get_tensors_from_graph(graph):
    args = dict( data='tensor' )
    if graph.is_multigraph():
        args['keys'] = True

    tensors = []
    for edgedata in graph.edges.data(**args):
        u, v, key, tensor = edgedata
        tensors.append(tensor)
    # use only unique
    tensors = ({id(t):t for t in tensors}).values()
    tensors = list(tensors)
    return tensors

def get_id(**args):
    id = "p{p}_expr.S{S}_{type}_d{degree}_seed{seed}".format(**args)
    return id

def as_json(size
            , qaoa_layers=1
            , type='randomreg', degree=3, seed=42
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
        graph_repr = nx.node_link_data(G)
        to_db =  {}
        # TODO: generator of id should be separate
        to_db['_id'] = get_id(**args)
        # Note: mongodb will try to use all the nested dicts,
        #       so store the graph as string
        to_db['tensors'] = get_tensors_from_graph(G)

        for t in to_db['tensors']:
            del t['data_key']

        to_db['graph'] = graph_repr
        to_db['n_qubits'] = n_qubits
        to_db['extra'] = args
        to_db['tags'] = ['qaoa', 'maxCut', 'expr']
        str = json.dumps(to_db)
        print(str)

    else:
        raise Exception(f"Invalid graph type {type}, should be one of {types_allowed}")

if __name__ == "__main__":
    fire.Fire(as_json)

