import numpy as np
import tensornetwork as tn
import xyzpy as xyz
from gates import xgate, cnot, hgate
from mps import MPS

def test_from_wavefunction_all_zero_state():
    wavefunction = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    mps = MPS.construct_mps_from_wavefunction(wavefunction, 'q', 3, 2)

    assert mps.get_norm() == 1
    assert isinstance(mps, MPS)
    assert mps.N == 3
    assert mps.physical_dim == 2
    assert np.allclose(mps.get_wavefunction(), wavefunction)

def test_from_wavefunction_random():
    n = 3
    wavefunction = np.random.rand(2**n)
    wavefunction = wavefunction / np.linalg.norm(wavefunction, ord = 2)
    mps = MPS.construct_mps_from_wavefunction(wavefunction, 'q', n, 2)
    assert mps.get_norm() == 1
    assert np.allclose(mps.get_wavefunction(), wavefunction)

def test_apply_one_qubit_mps_operation_xgate():
    # q0q1 = 00
    # On apply x gate |00> -> |10>
    mps = MPS("q", 2, 2)
    assert mps.get_norm() == 1
    
    mps.apply_single_qubit_gate(xgate(), 0)
    assert mps.get_norm() == 1

    assert np.allclose(mps.get_wavefunction(), np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex64))

def test_apply_twoq_cnot_two_qubits():
    # In the following tests, the first qubit is always the control qubit.
    # Check that CNOT|10> = |11>
    mps = MPS("q", 2, 2)
    assert mps.get_norm() == 1

    mps.apply_single_qubit_gate(xgate(), 0)
    mps.apply_two_qubit_gate(cnot(), [0, 1])
    assert mps.get_norm() == 1
    assert np.allclose(mps.get_wavefunction(), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex64))

def test_apply_gate_for_bell_circuit():
    mps = MPS("q", 2, 2)
    assert mps.get_norm() == 1

    mps.apply_single_qubit_gate(hgate(), 0)
    mps.apply_two_qubit_gate(cnot(), [0,1])
    assert mps.get_norm() == 1
    assert np.allclose(mps.get_wavefunction(), np.array([0.707, 0.0, 0.0, 0.707], dtype=np.complex64))

def test_apply_gate_for_ghz_circuit():
    mps = MPS("q", 3, 2)
    assert mps.get_norm() == 1

    mps.apply_single_qubit_gate(hgate(), 0)
    mps.apply_two_qubit_gate(cnot(), [0,1])
    mps.apply_two_qubit_gate(cnot(), [1,2])
    assert mps.get_norm() == 1
    assert np.allclose(mps.get_wavefunction(), np.array([ 0.7071, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7071], dtype=np.complex64))


test_apply_one_qubit_mps_operation_xgate()