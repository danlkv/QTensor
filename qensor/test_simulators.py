from qensor import CirqQAOAComposer, QtreeQAOAComposer
from qensor.Simulate import CirqSimulator, QtreeSimulator
import numpy as np
import networkx as nx


def get_test_problem():
    w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
    G = nx.from_numpy_matrix(w)
    G = nx.random_regular_graph(3, 10)
    gamma, beta = [np.pi/3], [np.pi/2]
    return G, gamma, beta

def test_qtree():
    G, gamma, beta = get_test_problem()

    composer = QtreeQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    composer.ansatz_state()

    print(composer.circuit)
    sim = QtreeSimulator()
    result = sim.simulate(composer.circuit)
    print(result.data)
    qtree_amp = result.data

    composer = CirqQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    composer.ansatz_state()

    print(composer.circuit)
    sim = CirqSimulator()
    result = sim.simulate(composer.circuit)
    print(result)
    final_cirq = result.final_state
    assert final_cirq[0] - qtree_amp < 1e-5

    assert result

