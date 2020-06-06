import sys
sys.path.append('..')
sys.path.append('./qaoa')

import pyrofiler as prof
from multiprocessing.dummy import Pool
from multiprocessing import Pool
import utils_qaoa as qaoa
import utils
import numpy as np
import qtree
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def just_remove_peo(x):
    graph, peo = x
    peo = [var for var in peo if var in graph.nodes]
    return peo


def get_qbb_peo(graph):
    try:
        peo, tw = qtree.graph_model.get_peo(graph)
        fail = False
    except TypeError as e:
        print(repr(e), e)
        print('QBB fail, nodes count:', graph.number_of_nodes())
        peo, nghs = utils.get_locale_peo(graph, utils.n_neighbors)
        fail = True
    return peo, fail

def contract_by_peo(old_graph, peo):
    graph = copy.deepcopy(old_graph)
    for n in peo:
        qtree.graph_model.eliminate_node(graph, n)
    return graph

def get_chop_idxs(graph, peo, cost, nghs):
    drop_idx = get_chop_dn_drop(nghs)
    min_idx = np.argmin(cost[0][:drop_idx])
    before_min = min_idx - (drop_idx-min_idx)
    on_plato = 2 * min_idx // 3
    idxs = [on_plato,  before_min, min_idx, min_idx+5]
    idxs += [drop_idx-15, drop_idx-5, drop_idx-2, drop_idx-1]
    idxs += [drop_idx, drop_idx + 1, drop_idx + 2, drop_idx+5]

    return idxs

def cost_before_chop(idxs, cost):
    mems, floats = cost
    before_mem = [max(mems[:i]) for i in idxs]
    return before_mem

def get_chop_dn_drop(nghs):
    nghs = np.array(nghs)
    dn = nghs[1:] - nghs[:-1]
    neg_idx = [i for i, n in enumerate(dn) if n<0]
    pos_idx = [i for i, n in enumerate(dn) if n>0]
    drop_idx = neg_idx[0]
    pos_idx.reverse()
    before_drop = [i for i in pos_idx if i<drop_idx]
    return before_drop[0] - 1
def n_peo(graph):
    return utils.get_locale_peo(graph, utils.n_neighbors)
def get_cost(x):
    return qaoa.cost_graph_peo(*x)
