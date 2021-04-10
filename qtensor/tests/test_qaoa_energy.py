import numpy as np
import networkx as nx
import pytest
from functools import lru_cache

import pyrofiler as prof

from qtensor import CirqQAOAComposer, QtreeQAOAComposer, ZZQtreeQAOAComposer, DefaultQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.Simulate import CirqSimulator, QtreeSimulator
from qtensor.FeynmanSimulator import FeynmanSimulator
from qtensor.optimisation.Optimizer import TamakiTrimSlicing, TreeTrimSplitter
from qtensor.tests.qiskit_qaoa_energy import simulate_qiskit_amps
from qtensor.tests import get_test_problem


@pytest.fixture
def test_problem(request):
    n, p, d, type = request.param
    return get_test_problem(n, p, d, type)


paramtest = [
    # n, p, degree, type
     [4, 4, 3, 'random']
    ,[10, 5, 2, 'random']
    ,[14, 1, 3, 'random']
    ,[3, 3, 0, 'grid2d']
    ,[8, 4, 0, 'line']
]

@pytest.mark.parametrize('test_problem', paramtest ,indirect=True)
def test_default_qaoa_energy_vs_qiskit(test_problem):
    G, gamma, beta = test_problem
    print('default', G.number_of_nodes(), G.number_of_edges(), len(gamma))
    sim = QAOAQtreeSimulator(DefaultQAOAComposer)
    with prof.timing('QTensor energy time'):
        E = sim.energy_expectation(G, gamma=gamma, beta=beta)
    assert E

    gamma, beta = -np.array(gamma)*2*np.pi, np.array(beta)*np.pi
    with prof.timing('Qiskit energy time'):
        qiskit_E = simulate_qiskit_amps(G, gamma, beta)
    assert np.isclose(E, qiskit_E)

@pytest.mark.parametrize('test_problem', paramtest ,indirect=True)
def test_CC_qaoa_energy_vs_qiskit(test_problem):
    G, gamma, beta = test_problem
    print('cc', G.number_of_nodes(), G.number_of_edges(), len(gamma))
    sim = QAOAQtreeSimulator(ZZQtreeQAOAComposer)
    with prof.timing('QTensor energy time'):
        E = sim.energy_expectation(G, gamma=gamma, beta=beta)
    assert E

    gamma, beta = -np.array(gamma)*2*np.pi, np.array(beta)*np.pi
    with prof.timing('Qiskit energy time'):
        qiskit_E = simulate_qiskit_amps(G, gamma, beta)
    assert np.isclose(E, qiskit_E)


@pytest.mark.parametrize('test_problem', paramtest ,indirect=True)
def test_qaoa_energy_vs_qiskit(test_problem):
    G, gamma, beta = test_problem
    print('no_zz', G.number_of_nodes(), G.number_of_edges(), len(gamma))
    sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    with prof.timing('QTensor energy time'):
        E = sim.energy_expectation(G, gamma=gamma, beta=beta)
    assert E

    gamma, beta = -np.array(gamma)*2*np.pi, np.array(beta)*np.pi
    with prof.timing('Qiskit energy time'):
        qiskit_E = simulate_qiskit_amps(G, gamma, beta)
    assert np.isclose(E, qiskit_E)

def test_qaoa_energy_multithread():
    G, gamma, beta = get_test_problem()
    sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    res = sim.energy_expectation_parallel(
        G, gamma=gamma, beta=beta,
        n_processes=4
    )
    print('result parallel', res)
    assert res
    res_1 = sim.energy_expectation(
        G, gamma=gamma, beta=beta)
    print('result serial', res_1)
    assert res_1 - res < 1e-6

class FeynmanQAOASimulator(QAOAQtreeSimulator, FeynmanSimulator):
    pass

def test_qaoa_energy_feynman():
    G, gamma, beta = get_test_problem(10, 3, 3)
    sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    res = sim.energy_expectation(
        G, gamma=gamma, beta=beta)
    print('result simple simulator', res)

    sim = FeynmanQAOASimulator(QtreeQAOAComposer)
    sim.optimizer = TreeTrimSplitter(max_tw=13, tw_bias=0)
    res_1 = sim.energy_expectation(
        G, gamma=gamma, beta=beta)
    print('result feynman simulator', res_1)
    assert np.isclose(res, res_1)

if __name__ == '__main__':
    test_qaoa_energy_multithread()
