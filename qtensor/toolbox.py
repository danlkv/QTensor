import networkx as nx
import numpy as np
from tqdm import tqdm
import time

from qensor.optimisation.TensorNet import QtreeTensorNet
from qensor.optimisation.Optimizer import OrderingOptimizer, TamakiOptimizer, WithoutOptimizer
from qensor import QtreeQAOAComposer

def qaoa_energy_tw_from_graph(G, p, max_time=0, max_tw=0, ordering_algo='greedy'):
    gamma, beta = [0]*p, [0]*p
    def get_tw(circ):

        tn = QtreeTensorNet.from_qtree_gates(circ)

        if ordering_algo=='greedy':
            opt = OrderingOptimizer()
        elif ordering_algo=='tamaki':
            opt = TamakiOptimizer(wait_time=45)
        elif ordering_algo=='without':
            opt = WithoutOptimizer()
        else:
            raise ValueError("Ordering algorithm not supported")
        peo, tn = opt.optimize(tn)
        treewidth = opt.treewidth
        return treewidth

    twidths = []
    if max_time:
        start = time.time()
    else:
        start = np.inf
    for edge in tqdm(G.edges()):
        composer = QtreeQAOAComposer(G, beta=beta, gamma=gamma)
        composer.energy_expectation_lightcone(edge)
        tw = get_tw(composer.circuit)
        if max_tw:
            if tw>max_tw:
                print(f'Encountered treewidth of {tw}, which is larger {max_tw}')
                break
        twidths.append(tw)
        if time.time() - start > max_time:
            break
    print(f'med={np.median(twidths)} mean={round(np.mean(twidths), 2)} max={np.max(twidths)}')
