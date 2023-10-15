import numpy as np
import tensornetwork as tn
import xyzpy as xyz
from mps_operation import xgate, cnot, hgate
from mps import MPS

def test_from_wavefunction_all_zero_state():
    wavefunction = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    mps = MPS.construct_mps_from_wavefunction(wavefunction, 'q', 3, 2)

    # kwargs.setdefault("legend", True)
    # kwargs.setdefault("compass", True)
    # kwargs.setdefault("compass_labels", mps.inds)
    # xyz.visualize_tensor(mps.get_tensors(False), legend=True, compass=True)

    assert isinstance(mps, MPS)
    assert mps.N == 3
    assert mps.physical_dim == 2
    assert np.allclose(mps.get_wavefunction(), wavefunction)

def test_from_wavefunction_random():
    n = 3
    wavefunction = np.random.rand(2**n)
    wavefunction = wavefunction / np.linalg.norm(wavefunction, ord = 2)
    mps = MPS.construct_mps_from_wavefunction(wavefunction, 'q', n, 2)
    assert np.allclose(mps.get_wavefunction(), wavefunction)
#def add random function
#  

# Entangled

def test_apply_one_qubit_mps_operation_xgate():
    mps = MPS("q", 2, 2)
    print("Before applying gate: ", mps.get_wavefunction())

    # nodes = mps.get_tensors(False)
    # qubits = [node[0] for node in nodes]
    # q0q1 = 00
    # On apply x gate |00> -> |10>
    mps.apply_single_qubit_gate(xgate(), 0)
    
    print("After applying x gate: ", mps.get_wavefunction())

def test_apply_twoq_cnot_two_qubits():
    """Tests for correctness of final wavefunction after applying a CNOT
    to a two-qubit MPS.
    """
    # In the following tests, the first qubit is always the control qubit.
    # Check that CNOT|10> = |11>
    mps = MPS("q", 2, 2)
    mps.apply_single_qubit_gate(xgate(), 0)

    print("After applying x gate: ", mps.get_wavefunction())

    mps.apply_two_qubit_gate(cnot(), [0, 1])
    print("After applying cnot gate: ", mps.get_wavefunction())


    correct = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex64)
    curr = mps.get_wavefunction()
    # assert np.allclose(curr, correct)

def test_apply_gate_for_bell_circuit():
    mps = MPS("q", 2, 2)
    mps.apply_single_qubit_gate(hgate(), 0)
    mps.apply_two_qubit_gate(cnot(), [0,1])

    print("Wavefunction for bell circuit: ", mps.get_wavefunction())

def test_apply_gate_for_ghz_circuit():
    mps = MPS("q", 3, 2)
    mps.apply_single_qubit_gate(hgate(), 0)
    mps.apply_two_qubit_gate(cnot(), [0,1])
    mps.apply_two_qubit_gate(cnot(), [1,2])

    print("Wavefunction for ghz circuit: ", mps.get_wavefunction())

# test_from_wavefunction_all_zero_state()
# test_apply_one_qubit_mps_operation_xgate()
# test_from_wavefunction_random()

#test_apply_twoq_cnot_two_qubits()

# test_apply_gate_for_bell_circuit()
test_apply_gate_for_ghz_circuit()