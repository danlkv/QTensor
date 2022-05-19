from qtensor.ext import cuquantum as cq
from qtensor.tests import get_test_problem
from qtensor import DefaultQAOAComposer, QtreeSimulator
import random, numpy as np

random.seed(10)
np.random.seed(10)


import logging
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s %(levelname)-8s %(message)s', force=True )

def test_cuquantum_tn():
    G, gamma, beta = get_test_problem(16, 3, d=3)
    composer = DefaultQAOAComposer(
            graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    circ = composer.circuit
    print('circ', circ)
    print('circ len', len(circ))
    tn = cq.CuTensorNet.from_qtree_gates(circ)
    slicer_opt = cq.cq.SlicerOptions()
    reconf_opt = cq.cq.ReconfigOptions(num_iterations=0)   
    print('Cuquantum equation', tn._eq)
    print('Cuquantum tensor count', len(tn))
    path, info = tn.net.contract_path(
        optimize=dict(
            samples=16,
            threads=1,
            slicing=slicer_opt,
            reconfiguration=reconf_opt
        )
    )

def test_cuquantum_opt():
    G, gamma, beta = get_test_problem(16, 3, d=3)
    composer = DefaultQAOAComposer(
            graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    circ = composer.circuit
    tn = cq.CuTensorNet.from_qtree_gates(circ)
    slicer_opt = cq.cq.SlicerOptions()
    print('Cuquantum equation', tn._eq)
    print('Cuquantum tensor count', len(tn))
    opt = cq.CuQuantumOptimizer(slicing=slicer_opt, threads=1, samples=10)
    path, tn = opt.optimize(tn)
    print('Width cuquantum', opt.treewidth)

def test_cuquantum_sim():
    G, gamma, beta = get_test_problem(16, 4, d=3)
    composer = DefaultQAOAComposer(
            graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    circ = composer.circuit
    sim = cq.CuQuantumSimulator()
    res = sim.simulate(circ)

    sim2 = QtreeSimulator()
    res2 = sim2.simulate(circ)
    assert np.allclose(res, res2)

def test_cuquantum_sim_batch():
    G, gamma, beta = get_test_problem(16, 4, d=3)
    composer = DefaultQAOAComposer(
            graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    circ = composer.circuit
    sim = cq.CuQuantumSimulator()
    res = sim.simulate_batch(circ, batch_vars=6)

    sim2 = QtreeSimulator()
    res2 = sim2.simulate_batch(circ, batch_vars=6)
    assert np.allclose(res.flatten(), res2)
