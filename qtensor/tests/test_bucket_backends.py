import numpy as np
import networkx as nx
import pytest

from qtensor import QtreeQAOAComposer
from qtensor.Simulate import CirqSimulator, QtreeSimulator

from qtensor.ProcessingFrameworks import PerfBackend
from qtensor.ProcessingFrameworks import PerfNumpyBackend

from qtensor.ProcessingFrameworks import CMKLExtendedBackend


@pytest.fixture(scope="module")
def test_problem():
    w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
    G = nx.from_numpy_matrix(w)

    G = nx.random_regular_graph(3, 18)
    gamma, beta = [np.pi/3]*2, [np.pi/2]*2
    yield G, gamma, beta

@pytest.fixture(scope='module')
def ground_truth_energy(test_problem):
    G, gamma, beta = test_problem
    composer = QtreeQAOAComposer(graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    sim = QtreeSimulator()

    result = sim.simulate(composer.circuit)
    yield result


def test_profiled(capsys, ground_truth_energy, test_problem):
    G, gamma, beta = test_problem
    composer = QtreeQAOAComposer(graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    backend = PerfNumpyBackend()
    sim = QtreeSimulator(bucket_backend=backend)

    result = sim.simulate(composer.circuit)
    print("Profile results")
    print(backend.gen_report())

    assert np.allclose(result, ground_truth_energy)

def test_mkl(capsys, test_problem, ground_truth_energy):
    G, gamma, beta = test_problem
    composer = QtreeQAOAComposer(graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    backend = PerfBackend.from_backend(CMKLExtendedBackend, print=False)
    sim = QtreeSimulator(bucket_backend=backend)

    result = sim.simulate(composer.circuit)
    print("Profile results")
    print(backend.gen_report())

    assert np.allclose(result, ground_truth_energy)


if __name__=='__main__':
    test_profiled(None)
