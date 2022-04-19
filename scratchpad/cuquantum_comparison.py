import cuquantum as cq
import qtensor as qt
import quimb as qu
import networkx as nx
import time
import re

from quimb.tensor.tensor_core import concat, _gen_output_inds, _inds_to_eq

N = 32
p = 2
gamma, beta = [.2]*p, [.3]*p
G = nx.random_regular_graph(3, N)


print('Optimization step:')

print('Quimb = ', end='', flush=True)
# -- Quimb
terms = {(i, j): 1 for i,j in G.edges}
qu_c = qu.tensor.circuit_gen.circ_qaoa(terms, p, gamma, beta)
start = time.time()
qu_r = qu_c.amplitude_rehearse()
end = time.time()

print(qu_r['W'])

print('Quimb path finding time=', end-start)
# -- QTensor

print('QTensor = ', end='', flush=True)
comp = qt.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
comp.ansatz_state()
tn = qt.optimisation.QtreeTensorNet.from_qtree_gates(comp.circuit)
opt = qt.toolbox.get_ordering_algo('rgreedy_0.02_20')
start = time.time()
peo, tw = opt.optimize(tn)
end = time.time()
print(opt.treewidth)
print('QTensor path finding time=', end-start)

print('CuQuantum FLOPS = ', end='', flush=True)
# -- CuQuantum
info = qu_r['info']
qu_tn = qu_r['tn']
eq = info.eq
tdata = [t.data for t in qu_tn.tensors]

# ---- CuQuantum with hyperoptimizer
# Turn cuquantum logging on.
import logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', force=True )

# Compute the path.
# 1. Disable slicing for this problem.
# 2. Use 16 hyperoptimizer samples. 
start = time.time()
slicer_opt = cq.SlicerOptions(disable_slicing=True)
path, info = cq.contract_path(eq, *tdata, optimize={'samples' : 16, 'slicing': slicer_opt})
end = time.time()

print(f"{info.opt_cost}, largest intermediate = {info.largest_intermediate} Elements")
print('CuQuantum hyper path finding time = ', end-start)

# ---- Vanilia Cuquantum
cq_r = cq.einsum_path(eq, *tdata)

#el =  re.search('Cuquantum vanilia: Largest intermediate = ?(.+)', cq_r[1]).groups()[0]
#print(el)

print('Simulation:')

qu_c = qu.tensor.circuit_gen.circ_qaoa(terms, p, gamma, beta)
print('Quimb = ', end='', flush=True)

start = time.time()
qu_r = qu_c.amplitude(''.join(['0']*N))
end = time.time()
print(end-start)


print('QTensor *contraction only* time = ', end='', flush=True)
comp = qt.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
comp.ansatz_state()
sim = qt.QtreeSimulator(optimizer=opt)
start = time.time()
sim.simulate_batch(comp.circuit, peo=peo)
end = time.time()

print(end - start)

#print('CuQuantum = ', end='', flush=True)
#cq_r = cq.einsum(eq, *tdata)
print('CuQuantum *contraction only* time = ', end='', flush=True)
start = time.time()
cq_r = cq.contract(eq, *tdata, optimize={'path' : path})
end = time.time()
print(end - start)
