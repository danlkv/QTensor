from qtensor import QtreeQAOAComposer
from qtensor.ProcessingFrameworks import PerfNumpyBackend

from qtensor.Simulate import CirqSimulator, QtreeSimulator
import numpy as np
import networkx as nx


def get_test_problem():
    w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
    G = nx.from_numpy_matrix(w)

    G = nx.random_regular_graph(5, 14)
    gamma, beta = [np.pi/3], [np.pi/2]
    return G, gamma, beta

def test_profiled(capsys):
    G, gamma, beta = get_test_problem()

    composer = QtreeQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    composer.ansatz_state()

    print(composer.circuit)
    backend = PerfNumpyBackend()
    sim = QtreeSimulator(bucket_backend=backend)

    result = sim.simulate(composer.circuit)
    print("Profile results")
    print(backend.gen_report())

    qtree_amp = result

    assert qtree_amp
