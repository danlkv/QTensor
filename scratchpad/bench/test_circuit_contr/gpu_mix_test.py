import qtensor as qt
import networkx as nx
import time
import re
import random
import numpy as np
SEED = 19
random.seed(SEED)
np.random.seed(SEED)

N = 38
p = 4
gamma, beta = [.2]*p, [.3]*p
G = nx.random_regular_graph(3, N)

print('Optimization step:')

print('QTensor = ', end='', flush=True)
comp = qt.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
comp.ansatz_state()
tn = qt.optimisation.QtreeTensorNet.from_qtree_gates(comp.circuit)
opt = qt.toolbox.get_ordering_algo('tamaki_7')
start = time.time()
peo, tw = opt.optimize(tn)
end = time.time()
print(opt.treewidth)
print('QTensor path finding time=', end-start)

print('Simulation:')

print('QTensor *contraction only* GPU time = ', end='', flush=True)
comp = qt.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
comp.ansatz_state()
backend = qt.contraction_backends.get_mixed_backend('torch_cpu', 'torch_gpu', 12)
backend = qt.contraction_backends.get_mixed_backend('einsum', 'cupy', 12)

#backend=qt.contraction_backends.get_backend('torch_gpu')
sim = qt.QtreeSimulator(optimizer=opt, backend=backend)
start = time.time()
sim.simulate_batch(comp.circuit, peo=peo)
end = time.time()

print(end - start)
