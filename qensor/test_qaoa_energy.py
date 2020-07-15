from qensor import CirqQAOAComposer, QtreeQAOAComposer
from qensor import QAOAQtreeSimulator
from qensor.Simulate import CirqSimulator, QtreeSimulator
from qensor.FeynmanSimulator import FeynmanSimulator
from qensor.optimisation.Optimizer import TamakiTrimSlicing
import numpy as np
import networkx as nx

def get_test_problem(n=14, p=2, d=3):
    w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
    G = nx.from_numpy_matrix(w)

    G = nx.random_regular_graph(d, n)
    gamma, beta = [np.pi/3]*p, [np.pi/2]*p
    return G, gamma, beta

def test_qaoa_energy():
    G, gamma, beta = get_test_problem()
    sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    res = sim.energy_expectation(
        G, gamma=gamma, beta=beta)
    print('result', res)
    assert res

def test_qaoa_energy_multithread():
    G, gamma, beta = get_test_problem()
    sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    res = sim.energy_expectation_parallel(
        G, gamma=gamma, beta=beta,
        n_processes=4
    )
    print('result', res)
    assert res
    res_1 = sim.energy_expectation(
        G, gamma=gamma, beta=beta)
    print('result', res_1)
    assert res_1 - res < 1e-6

class FeynmanQAOASimulator(QAOAQtreeSimulator, FeynmanSimulator):
    pass

def test_qaoa_energy_feynman():
    G, gamma, beta = get_test_problem(30, 4, 3)
    sim = FeynmanQAOASimulator(QtreeQAOAComposer)
    sim.opt_args['tw_bias'] = 5
    sim.optimizer = TamakiTrimSlicing
    res_1 = sim.energy_expectation(
        G, gamma=gamma, beta=beta)
    print('result', res_1)

if __name__ == '__main__':
    test_qaoa_energy_multithread()
