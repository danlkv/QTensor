import json
import fire
import sys
import networkx as nx

import expr_optim
import expr
import utils_qaoa

def _iter_input_json():
    for x in sys.stdin:
        try:
            yield json.loads(x)
        except json.decoder.JSONDecodeError:
            continue

def read_expressions():
    for expr_data in _iter_input_json():
        # do not suppord dict_of_lists for now, but if do, will use tensors
        # tensors = expr_data['tensors']
        G = nx.node_link_graph(expr_data['graph'])
        G = nx.convert_node_labels_to_integers(G)
        yield G, expr_data['_id']



def as_json(read=True):
    """
    Return a JSON representation of optimized expression

    """
    if read:
        exprs = read_expressions()
    else:
        # fallback to default
        args = dict(S=9, p=1, type='randomreg', degree=3, seed=42)
        G, _  = utils_qaoa.get_test_expr_graph(**args)
        expr_id = expr.get_id(**args)
        exprs = ((G,expr_id), )

    for G, expr_id in exprs:
        print(G.nodes())

        peo, tw, G = expr_optim.optimize_order(G)

        to_db =  {}
        # TODO: generator of id should be separate
        to_db['_id'] = 'ordered_qbb.'+ expr_id
        to_db['expr_id'] = expr_id
        # Note: mongodb will try to use all the nested dicts,
        #       so store the graph as string
        to_db['tensors'] = expr.get_tensors_from_graph(G)
        for t in to_db['tensors']:
            try:
                del t['data_key']
            except KeyError:
                pass

        to_db['transforms'] =[{
            'type':'order',
            'params': {'order':peo, 'tw':tw, 'algo':'QuickBB'}
        }]

        str = json.dumps(to_db)
        print(str)

if __name__ == "__main__":
    fire.Fire(as_json)

