from .CircuitComposer import QAOAComposer
from .OpFactory import CirqCreator, QtreeCreator

import networkx as nx
import numpy as np

import cirq

class CirqQAOAComposer(QAOAComposer, CirqCreator):
    pass

class QtreeQAOAComposer(QAOAComposer, QtreeCreator):
    pass

def get_test_problem():
    w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
    G = nx.from_numpy_matrix(w)
    gamma, beta = [np.pi/3], [np.pi/2]
    return G, gamma, beta


def test_cirq():
    G, gamma, beta = get_test_problem()

    composer = CirqQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    composer.anzatz_state()
    sim = cirq.Simulator()
    result = sim.simulate(composer.circuit)

    print(result)
    assert result


