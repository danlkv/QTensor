import joblib
import sys
sys.path.append('..')
import utils_qaoa as qaoa
import qtree
import utils

memory = joblib.Memory('./qaoa/cached_data/memcache')

@memory.cache
def qaoa_expr_graph(size, p=1, type='grid', seed=42, **kw):
    return qaoa.get_test_expr_graph(size, p, type=type, seed=seed, **kw)

def neigh_peo(size, p=1, type='grid',**kw):
    graph, N = memory.cache(qaoa_expr_graph)(size, p, type=type, **kw)
    peo, nghs = utils.get_locale_peo(graph, utils.n_neighbors)
    return peo, nghs

def graph_contraction_costs(size, peo, p=1, type='grid', **kw):
    graph_old, N = memory.cache(qaoa_expr_graph)(size, p, type=type, **kw)
    peo, nghs = memory.cache(neigh_peo)(size, p, type=type, **kw)
    graph, _ = utils.reorder_graph(graph_old, peo)
    costs = qtree.graph_model.cost_estimator(graph)
    return costs

@memory.cache
def contracted_graph(size, peo, idx, p=1, type='grid', **kw):
    graph, N = memory.cache(qaoa_expr_graph)(size, p, type=type, **kw)
    peo, nghs = memory.cache(neigh_peo)(size, p, type=type, **kw)
    for n in peo[:idx]:
        qtree.graph_model.eliminate_node(graph, n)
    return graph
