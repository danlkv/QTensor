import sys
sys.path.append('..')
sys.path.append('./qaoa')

import pyrofiler as prof
import utils_qaoa as qaoa
import utils
import numpy as np

import qtree

import logging
logging.disable(30)

from loguru import logger as log
log.remove()

class Data:
    def _pitr(self):
        for k, value  in self.__dict__.items():
            if k[0]=='_':continue
            yield k, value

    def print(self):
        for k, value in self._pitr():
            print(f'{k}\t', value)

    def _shorten_str(self, s, T=70):
        L = len(s)
        if L>T:
            s = s[:T//2]+'[...]'+s[-T//2:]
        return s

    def print_shortened(self):
        for k, value  in self._pitr():
            s = str(value)
            print(f'{k}\t', self._shorten_str(s))


######### #########

data = Data()
S = int(sys.argv[1])
seed = int(sys.argv[2])

qc, n_qubits = qaoa.get_test_qaoa(S, p=1, type='randomreg'
                                  , seed=seed, degree=3)

bck, data_dict, bra, ket = qtree.optimizer.circ2buckets(n_qubits, qc)
graph = qtree.graph_model.buckets2graph(bck)
#graph, n_qubits = qaoa.get_test_expr_graph(S, p=1, type='randomreg' , seed=seed, degree=3)

data.n_qubits = n_qubits
L = graph.number_of_nodes()
data.nodes_in_graph = L

qbb = False
if qbb:
    peo, tw = qtree.graph_model.get_peo(graph)
    peo = [int(x) for x in peo]
    nghs = utils.get_neighbours_path(graph, peo=peo)
    #print(list(zip(range(len(peo)), peo, nghs)))
    imax = len(peo) - max(nghs)
else:
    peo, nghs = utils.get_locale_peo(graph, utils.n_neighbors)
    imax = np.argmax(nghs)

data.peo = peo
data.neighbors_path = nghs

assert len(data.peo) == L

data.max_nghs = max(nghs)

data.index_of_max_nghs = imax

data.print()
print('='*10)
data.print_shortened()


## Find optimal parallelization point

def nghs_with_par(graph, par_rank):
    pvars, graph_opt = qtree.graph_model.split_graph_by_metric(graph, par_rank)
    peo, nghs = utils.get_locale_peo(graph_opt, utils.n_neighbors)
    return max(nghs), pvars

residue = L - imax
# Search for different points
lower_search_pt = max(0, L - 3*residue)
upper_search_pt = imax
print('Idx\tresidue\tmax_nghs\tcurrent_ngh\tpvars')

for n in peo[:lower_search_pt]:
    qtree.graph_model.eliminate_node(graph, n)

par_rank = int(sys.argv[3])

def print_row(idx, val, pvars):
    print(f'{idx}\t{L-idx}\t{val}\t{nghs[idx]}\t{pvars}')

stats={}
for I in range(lower_search_pt, upper_search_pt +1):
    qtree.graph_model.eliminate_node(graph, peo[I])
    width, pvars = nghs_with_par(graph, par_rank)
    print_row(I,width, pvars)
    stats[I]=width

revmap = {v:k for k,v in reversed(list(stats.items()))}
print('Revmap', revmap)


