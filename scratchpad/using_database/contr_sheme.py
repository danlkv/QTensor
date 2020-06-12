import json
import fire
import sys
import time
import networkx as nx
import sys
sys.path.append('..')

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



def as_json(read=True
            , ordering='qbb'
           ):
    """
    Return a JSON representation of optimized expression

    """
    if ordering not in ['qbb', 'nghs']:
        raise Exception("ordering should be one of 'nghs', 'qbb'")
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

        
        start_t = time.time()
        peo, tw, G = expr_optim.optimize_order(G, ordering_algo=ordering)
        end_t = time.time()

        to_db =  {}
        # TODO: generator of id should be separate
        to_db['_id'] = f'ordered_{ordering}.'+ expr_id
        to_db['expr_id'] = expr_id
        # Note: mongodb will try to use all the nested dicts,
        #       so store the graph as string
        to_db['tensors'] = expr.get_tensors_from_graph(G)
        for t in to_db['tensors']:
            try:
                del t['data_key']
            except KeyError:
                pass

        def get_machine_id():
            import socket
            import getpass
            host = socket.gethostname()
            user = getpass.getuser()
            return user + '@' + host

        to_db['transforms'] =[{
            'type':'order'
            ,'contract': {'order':peo}
            ,'extra': {'tw':tw, 'algo': ordering
                       ,'perf':{'time':end_t-start_t, 'run_by':get_machine_id()}
                      }
        }]

        str = json.dumps(to_db)
        print(str)

if __name__ == "__main__":
    fire.Fire(as_json)

