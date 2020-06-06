import sys
sys.path.append('..')
sys.path.append('./qaoa')

import pyrofiler as prof
import utils_qaoa as qaoa
import utils

import qtree

S = int(sys.argv[1])
seed = int(sys.argv[2])


graph, n_qubits = qaoa.get_test_expr_graph(S, p=1, type='randomreg'
                                 , seed=seed, degree=3)
print('n_qubits', n_qubits)

peo, tw = qtree.graph_model.get_peo(graph)
graph,_ = utils.reorder_graph(graph, peo)
nodes, nghs = utils.get_neighbours_path(graph)

print('peo', peo)
print('tw', tw)

print('nghs', nghs)
print(max(nghs))
