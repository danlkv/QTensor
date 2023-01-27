import qtensor
from qtensor import QtreeQAOAComposer
from qtensor.Simulate import QtreeSimulator
from qtensor.FeynmanSimulator import FeynmanSimulator

from qtensor.optimisation.Optimizer import WithoutOptimizer, TreeTrimSplitter, SlicesOptimizer
import numpy as np
import networkx as nx
from qtensor.tests import get_test_problem

np.random.seed(42)

def test_naive_slicing():
    N = 8
    G, gamma, beta = get_test_problem(n=N, p=8, d=3)
    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    sim = QtreeSimulator(optimizer=qtensor.toolbox.get_ordering_algo('naive'))
    reference = sim.simulate(composer.circuit)

    opt = TreeTrimSplitter(base_ordering='naive', max_tw=N, tw_bias=0)
    sim = FeynmanSimulator(optimizer=opt)
    result = sim.simulate(composer.circuit)
    print(result)
    assert np.allclose(result, reference)

    opts = [
        SlicesOptimizer(base_ordering='naive', max_tw=N, tw_bias=0, max_slice=6)
        , SlicesOptimizer(base_ordering='greedy', max_tw=N, tw_bias=0, max_slice=6)
        , SlicesOptimizer(base_ordering='rgreedy_0.02_10', max_tw=N, tw_bias=0, max_slice=6)
        , TreeTrimSplitter(base_ordering='greedy', max_tw=N, tw_bias=0, max_slice=6)
        , TreeTrimSplitter(base_ordering='rgreedy_0.02_10', max_tw=N, tw_bias=0, max_slice=6)
    ]
    tn = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(composer.circuit)
    for opt in opts:
        opt.optimize(tn)
        print("Width", opt.treewidth)
        if len(opt.parallel_vars) < 11:
            sim = FeynmanSimulator(optimizer=opt)
            result = sim.simulate(composer.circuit)
            print(result)
            assert np.allclose(result, reference)
        else:
            print(f"Skipping {opt} because it has too many ({len(opt.parallel_vars)}) parallel variables")

