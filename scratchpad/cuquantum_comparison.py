import cuquantum as cq
import qtensor as qt
import quimb as qu
import networkx as nx
import time
import re

from quimb.tensor.tensor_core import concat, _gen_output_inds, _inds_to_eq

N = 20
p = 4
gamma, beta = [.2]*p, [.3]*p
G = nx.random_regular_graph(3, N)


print('Optimization step:')

print('Quimb = ', end='', flush=True)
# -- Quimb
terms = {(i, j): 1 for i,j in G.edges}
qu_c = qu.tensor.circuit_gen.circ_qaoa(terms, p, gamma, beta)
qu_r = qu_c.amplitude_rehearse()

print(qu_r['W'])

# -- QTensor

print('QTensor = ', end='', flush=True)
comp = qt.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
comp.ansatz_state()
tn = qt.optimisation.QtreeTensorNet.from_qtree_gates(comp.circuit)
opt = qt.toolbox.get_ordering_algo('rgreedy_0.02_20')
opt.optimize(tn)
print(opt.treewidth)

print('CuQuantum = ', end='', flush=True)
# -- CuQuantum
info = qu_r['info']
qu_tn = qu_r['tn']
eq = info.eq
tdata = [t.data for t in qu_tn.tensors]

cq_r = cq.einsum_path(eq, *tdata)

el =  re.search('Largest intermediate = ?(.+)', cq_r[1]).groups()[0]
print(el)

print('Simulation:')

qu_c = qu.tensor.circuit_gen.circ_qaoa(terms, p, gamma, beta)
print('Quimb = ', end='', flush=True)

start = time.time()
qu_r = qu_c.amplitude(''.join(['0']*N))
end = time.time()
print(end-start)


print('QTensor = ', end='', flush=True)
comp = qt.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
comp.ansatz_state()
sim = qt.QtreeSimulator(optimizer=opt)
start = time.time()
sim.simulate(comp.circuit)
end = time.time()

print(end - start)

print('CuQuantum = ', end='', flush=True)
start = time.time()
cq_r = cq.einsum(eq, *tdata)
end = time.time()
print(end - start)
