from qtensor.optimisation.Optimizer import TamakiOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor import QtreeQAOAComposer
import networkx as nx

G = nx.random_regular_graph(3, 20)
p = 3
gamma, beta = [0.1]*p, [0.2]*p

composer = QtreeQAOAComposer(
	graph=G, gamma=gamma, beta=beta)
composer.ansatz_state()


tn = QtreeTensorNet.from_qtree_gates(composer.circuit)

opt = TamakiOptimizer(wait_time=15) # time in seconds for heuristic algorithm
peo, tn = opt.optimize(tn)
treewidth = opt.treewidth
print('Treewidth', treewidth)


