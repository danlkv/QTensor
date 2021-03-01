import numpy as np
import networkx as nx

from qtensor.optimisation.Optimizer import GreedyOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor import QtreeQAOAComposer

import qensor
import networkx as nx

def bethe_lattice(n):
    def add_two_nodes_to_leafs(graph):
        """ Works in-place """
        leaves = [n for n in graph.nodes() if graph.degree(n) <= 1]
        n = graph.number_of_nodes()
        for leaf in leaves:
            graph.add_edges_from([(leaf, n+1)
                                 ,(leaf, n+2)]
                               )
            n += 2
    graph = nx.Graph()
    graph.add_edges_from([(0,1)])
    for i in range(n):
        add_two_nodes_to_leafs(graph)
    return graph


D = 6
G = bethe_lattice(D)

for p in range(1, D+1):
    print(f'{p=}, {G.number_of_nodes()=}')
    gamma, beta = [0.1]*p, [0.2]*p
    composer = QtreeQAOAComposer(
            graph=G, gamma=gamma, beta=beta)
    composer.energy_expectation_lightcone((0,1))
    tn = QtreeTensorNet.from_qtree_gates(composer.circuit)
    print(f'{tn.get_line_graph().number_of_nodes()=}')

    opt = GreedyOptimizer()
    peo, tn = opt.optimize(tn)
    treewidth = opt.treewidth
    print(f"{treewidth=}")

