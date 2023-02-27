from qtensor import QtreeQAOAComposer
from qtensor.contraction_backends import PerfNumpyBackend

from qtensor.Simulate import CirqSimulator, QtreeSimulator
import qtensor
import numpy as np
import networkx as nx
from qtensor.tests import get_test_problem

from qtensor.contraction_algos import is_reverse_order_backend


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

def test_reverse_order_switch():
    backend = qtensor.contraction_backends.get_backend('torch')
    reverse = is_reverse_order_backend(backend)
    assert reverse

    backend = qtensor.contraction_backends.get_backend('einsum')
    reverse = is_reverse_order_backend(backend)
    assert not reverse

def test_compression_backend():
    pass
