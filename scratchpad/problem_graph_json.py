import fire
import json
import networkx as nx

import utils_qaoa


def as_json(size
            , type='randomreg', degree=3, seed=42
            , repr_way='dict_of_lists'
           ):

    """
    Return a JSON representation of a graph.

    repr_way: 'dict_of_lists' or 'dict_of_dicts'
    """
    types_allowed = ['randomreg', 'grid', 'rectgrid', 'randomgnp']
    if type in types_allowed:
        args = dict(S=size
                    , type=type
                    , degree=degree
                    , seed=seed
                   )
        G = utils_qaoa.get_test_graph(**args)
        #dict_ = nx.to_dict_of_dicts(G)
        if repr_way=='dict_of_dicts':
            dict_ = nx.to_dict_of_dicts(G)
        elif repr_way=='dict_of_lists':
            dict_ = nx.to_dict_of_lists(G)
        to_db =  {}
        # TODO: generator of id should be separate
        to_db['_id'] = f"S{size}_{type}_d{degree}_s{seed}"
        # Note: mongodb will try to use all the nested dicts,
        #       so store the graph as string
        to_db['graph'] = json.dumps(dict_)
        to_db['n_edges'] = G.number_of_edges()
        to_db['n_nodes'] = G.number_of_nodes()
        to_db['extra'] = args
        to_db['tags'] = ['qaoa', 'maxCut']
        str = json.dumps(to_db)
        print(str)

    else:
        raise Exception(f"Invalid graph type {type}, should be one of {types_allowed}")

fire.Fire(as_json)

