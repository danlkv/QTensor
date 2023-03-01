from qtensor import QtreeQAOAComposer
from qtensor.contraction_backends import PerfNumpyBackend
from qtensor.contraction_backends import CuPyBackend, NumpyBackend, CompressionBackend
from qtensor.compression import NumpyCompressor, CUSZCompressor
from qtensor.Simulate import CirqSimulator, QtreeSimulator

import pytest
import qtensor
import numpy as np
import networkx as nx
from qtensor.tests import get_test_problem, get_test_qaoa_ansatz_circ

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

ref_backend_name = 'cupy'
@pytest.mark.parametrize('circ', [
    get_test_qaoa_ansatz_circ(n=6, p=3),
    get_test_qaoa_ansatz_circ(n=12, p=4),
])
@pytest.mark.parametrize(['backend', 'atol'], [
    ('cupy', 1e-10),
    ('torch', 1e-10),
    ('cupy_compressed', 1e-10),
    (CompressionBackend(
        CuPyBackend(),
        CUSZCompressor(r2r_error=1e-4, r2r_threshold=1e-5),
        11 ),
        1e-5)
])
def test_backends(circ, backend, atol):
    ref_backend = qtensor.contraction_backends.get_backend(ref_backend_name)
    if isinstance(backend, str):
        backend = qtensor.contraction_backends.get_backend(backend)
    sim = QtreeSimulator(backend=backend)
    res = sim.simulate(circ)
    sim_ref = QtreeSimulator(backend=ref_backend)
    res_ref = sim_ref.simulate(circ)
    assert np.allclose(res, res_ref, atol=atol)
