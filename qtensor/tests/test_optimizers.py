import qtensor
from qtensor import CirqQAOAComposer, QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.optimisation.Optimizer import OrderingOptimizer, TamakiTrimSlicing, TreeTrimSplitter
from qtensor.optimisation.Optimizer import SlicesOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor.FeynmanSimulator import FeynmanSimulator
import numpy as np
import networkx as nx
np.random.seed(42)
import pytest

def get_test_problem(n=14, p=2, d=3):
    w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
    G = nx.from_numpy_matrix(w)

    G = nx.random_regular_graph(d, n)
    gamma, beta = [np.pi/3]*p, [np.pi/2]*p
    return G, gamma, beta

@pytest.mark.skip()
def test_tamaki_trimming_opt():
    G, gamma, beta = get_test_problem(34, p=3, d=3)

    composer = QtreeQAOAComposer(
            graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    sim = FeynmanSimulator()
    sim.optimizer = SlicesOptimizer
    result = sim.simulate(composer.circuit, batch_vars=3, tw_bias=7)
    print(result)


    sim.optimizer = TamakiTrimSlicing
    sim.opt_args = {'wait_time':5}
    result_tam = sim.simulate(composer.circuit, batch_vars=3, tw_bias=7)
    print(result_tam)
    assert np.allclose(result_tam , result)

if __name__ == '__main__':
    test_tamaki_trimming_opt()
