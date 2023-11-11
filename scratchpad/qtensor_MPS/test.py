import numpy as np
import tensornetwork as tn
import xyzpy as xyz
from gates import xgate, cnot, hgate, zgate
from mps import MPS


def test_from_wavefunction_all_zero_state():
    wavefunction = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    mps = MPS.construct_mps_from_wavefunction(wavefunction, "q", 3, 2)

    assert mps.get_norm() == 1
    assert isinstance(mps, MPS)
    assert mps.N == 3
    assert mps.physical_dim == 2
    assert np.allclose(mps.get_wavefunction(), wavefunction)


def test_from_wavefunction_random():
    n = 3
    wavefunction = np.random.rand(2**n)
    wavefunction = wavefunction / np.linalg.norm(wavefunction, ord=2)
    mps = MPS.construct_mps_from_wavefunction(wavefunction, "q", n, 2)
    assert np.isclose(mps.get_norm(), 1.0)
    assert np.allclose(mps.get_wavefunction(), wavefunction)


def test_apply_one_qubit_mps_operation_xgate():
    # q0q1 = 00
    # On apply x gate |00> -> |10>
    mps = MPS("q", 2, 2)
    assert np.isclose(mps.get_norm(), 1.0)

    mps.apply_single_qubit_gate(xgate(), 0)
    mps.apply_single_qubit_gate(xgate(), 1)
    assert np.isclose(mps.get_norm(), 1.0)

    assert np.allclose(
        mps.get_wavefunction(), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex64)
    )


def test_apply_twoq_cnot_two_qubits():
    # In the following tests, the first qubit is always the control qubit.
    # Check that CNOT|10> = |11>
    mps = MPS("q", 2, 2)
    assert np.isclose(mps.get_norm(), 1.0)

    mps.apply_single_qubit_gate(xgate(), 0)
    mps.apply_two_qubit_gate(cnot(), [0, 1])
    assert np.isclose(mps.get_norm(), 1.0)
    assert np.allclose(
        mps.get_wavefunction(), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex64)
    )


def test_apply_two_twoq_cnot_two_qubits():
    # In the following tests, the first qubit is always the control qubit.
    # Check that CNOT(0,1)|100> = |110>
    # Check that CNOT(1,2)|110> = |111>
    mps = MPS("q", 3, 2)
    assert np.isclose(mps.get_norm(), 1.0)

    mps.apply_single_qubit_gate(xgate(), 0)
    mps.apply_two_qubit_gate(cnot(), [0, 1])
    mps.apply_two_qubit_gate(cnot(), [1, 2])
    assert np.isclose(mps.get_norm(), 1.0)
    assert np.allclose(
        mps.get_wavefunction(),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.complex64),
    )


def test_apply_gate_for_bell_circuit():
    mps = MPS("q", 2, 2)
    assert np.isclose(mps.get_norm(), 1.0)

    mps.apply_single_qubit_gate(hgate(), 0)
    mps.apply_two_qubit_gate(cnot(), [0, 1])
    assert np.isclose(mps.get_norm(), 1.0)
    assert np.allclose(
        mps.get_wavefunction(),
        np.array([0.707106, 0.0, 0.0, 0.707106], dtype=np.complex64),
    )


def test_apply_gate_for_ghz_circuit():
    mps = MPS("q", 3, 2)
    assert np.isclose(mps.get_norm(), 1.0)

    mps.apply_single_qubit_gate(hgate(), 0)
    mps.apply_two_qubit_gate(cnot(), [0, 1])
    mps.apply_two_qubit_gate(cnot(), [1, 2])
    assert np.isclose(mps.get_norm(), 1.0)
    assert np.allclose(
        mps.get_wavefunction(),
        np.array([0.7071, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7071], dtype=np.complex64),
    )


def test_expectation_value_hgate():
    mps = MPS("q", 2, 2)
    copy = mps.__copy__()

    # <00|HI|00> = 1 / sqrt(2)
    np.isclose(mps.get_expectation(hgate(), 0), 1.0 / np.sqrt(2))
    assert mps.get_norm() == copy.get_norm()


def test_expectation_value_xgate():
    mps = MPS("q", 2, 2)
    copy = mps.__copy__()

    # <00|XI|00> = 0
    np.isclose(mps.get_expectation(xgate(), 0), 0.0)
    assert mps.get_norm() == copy.get_norm()


def test_expectation_value_xhgate():
    mps = MPS("q", 2, 2)
    copy = mps.__copy__()

    # <10|HI|10> = - 1 / sqrt(2)
    mps.apply_single_qubit_gate(xgate(), 0)
    np.isclose(mps.get_expectation(hgate(), 0), -1.0 / np.sqrt(2))
    assert mps.get_norm() == copy.get_norm()


def test_expectation_value_zgate():
    mps = MPS("q", 2, 2)
    copy = mps.__copy__()
    np.isclose(mps.get_expectation(zgate(), 0), 1.0)
    assert mps.get_norm() == copy.get_norm()


def test_expectation_value_xgate_at_k():
    k = 3
    n = 5
    mps = MPS("q", n, 2)
    copy = mps.__copy__()
    mps.apply_single_qubit_gate(xgate(), k)

    expectation_array = []

    for i in range(n):
        expectation_array.append(mps.get_expectation(zgate(), i))

    np.allclose(expectation_array, [1.0, 1.0, 1.0, -1.0, 1.0])
