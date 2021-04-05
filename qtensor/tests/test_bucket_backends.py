from qtensor import QtreeQAOAComposer
from qtensor.contraction_backends import PerfNumpyBackend

from qtensor.Simulate import CirqSimulator, QtreeSimulator
import numpy as np
import networkx as nx
from qtensor.tests import get_test_problem


def test_profiled(capsys):
    G, gamma, beta = get_test_problem()

    composer = QtreeQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    composer.ansatz_state()

    print(composer.circuit)
    backend = PerfNumpyBackend()
    sim = QtreeSimulator(backend=backend)

    result = sim.simulate(composer.circuit)
    print("Profile results")
    print(backend.gen_report())

    qtree_amp = result

    assert qtree_amp
