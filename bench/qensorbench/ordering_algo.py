import qtree
import networkx as nx
import sys
import glob
import numpy as np
from pathlib import Path

from qtensor import QtreeQAOAComposer
from qtensor.optimisation.Optimizer import GreedyOptimizer, TamakiOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet

def print_row(*args):
    row = [str(i) for i in args]
    print(','.join(row))

def get_test_problem(n=14, p=2, d=3):
    G = nx.random_regular_graph(d, n)
    gamma, beta = [np.pi/3]*p, [np.pi/2]*p
    return G, gamma, beta

def test_orderings():
    opt = GreedyOptimizer()
    tam = TamakiOptimizer(wait_time=5)
    seed = 43
    np.random.seed(seed)

    for n in range(14, 45, 2):
        p = 3
        G, gamma, beta = get_test_problem(n, p=p)
        composer = QtreeQAOAComposer(
                graph=G, gamma=gamma, beta=beta)
        composer.ansatz_state()

        tn = QtreeTensorNet.from_qtree_gates(composer.circuit)


        peo, tn = opt.optimize(tn)
        treewidth = opt.treewidth
        print_row(n, p, seed, 'greedy', treewidth)

        peo, tn = tam.optimize(tn)
        treewidth = tam.treewidth
        print_row(n, p, seed, 'tamaki', treewidth)

def test_orderings_bristlecone():
    opt = GreedyOptimizer()
    tam = TamakiOptimizer(wait_time=15)
    seed = 43
    np.random.seed(seed)

    brists = sys.argv[1]
    files = glob.glob(f'{brists}/*_0.txt')
    for filename in files:
        name = Path(filename).name
        n_qubits, circuit = qtree.operators.read_circuit_file(filename)
        circuit = sum(circuit, [])

        tn = QtreeTensorNet.from_qtree_gates(circuit)


        peo, tn = opt.optimize(tn)
        treewidth = opt.treewidth
        print_row(n_qubits, name, seed, 'greedy', treewidth)

        peo, tn = tam.optimize(tn)
        treewidth = tam.treewidth
        print_row(n_qubits, name, seed, 'tamaki', treewidth)

if __name__=='__main__':
    #test_orderings()
    test_orderings_bristlecone()
