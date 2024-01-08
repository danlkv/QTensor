import numpy as np
from gates import xgate, cnot, hgate, zgate, igate
from mps import MPS
from mpo import MPO, MPOLayer
import tensornetwork as tn
from constants import xmatrix, cnot_matrix


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
    mps = MPS("q", 4, 2)
    assert np.isclose(mps.get_norm(), 1.0)

    # mps.apply_single_qubit_gate(xgate(), 0)
    mps.apply_two_qubit_gate(cnot(), [1, 2])
    # assert np.isclose(mps.get_norm(), 1.0)
    # assert np.allclose(
    #     mps.get_wavefunction(), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex64)
    # )


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


def test_mps_mpo():
    mps00 = MPS("q", 2, 2)
    mps01 = MPS("q", 2, 2)
    mps10 = MPS("q", 2, 2)
    mps11 = MPS("q", 2, 2)

    mps01.apply_mpo(MPO(xgate(), [1]))
    mps10.apply_mpo(MPO(xgate(), [0]))
    mps11.apply_mpo(MPO(xgate(), [0]))
    mps11.apply_mpo(MPO(xgate(), [1]))

    allmps = (mps00, mps01, mps10, mps11)

    # Test inner products
    for i in range(4):
        for j in range(4):
            assert np.isclose(allmps[i].inner_product(allmps[j]), i == j)


def test_mps_single_qubit_mpo_layer():
    mps1 = MPS("q", 3, 2)
    mps1.apply_single_qubit_gate(xgate(), 1)
    assert mps1.get_norm() == 1
    # expectation = mps1(original) inner prod mps1(modified)
    mps2 = MPS("q", 3, 2)
    mpo_layer = MPOLayer("q", 3, 2)
    mpo_layer.add_single_qubit_gate(xmatrix, 1)
    mps2.apply_mpo_layer(mpo_layer)
    assert mps2.get_norm() == 1

    assert (mps1.get_wavefunction() == mps2.get_wavefunction()).all()


def test_mpo_two_qubit():
    mpo_layer = MPOLayer("q", 4, 2)
    # print("BEFORE")
    # print(mpo_layer._nodes)
    mpo_layer.add_two_qubit_gate(cnot(), [1, 2])
    print("AFTER")
    print(mpo_layer._nodes)
    # assert (mps1.get_wavefunction() == mps2.get_wavefunction()).all()


def test_mpo_construction_from_gate_function():
    # IZI
    I = np.eye(2)
    Z = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
    H = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex64)
    gate_func = np.kron(I, I)
    print(gate_func.shape)
    mpo = MPOLayer("q", 2, 2)
    mpo.construct_mpo(gate_func, "q", 2, 2)
    mps = MPS("q", 2, 2)
    print(mps.get_wavefunction())
    mps.apply_mpo_layer(mpo)
    print(mps.get_wavefunction())


# TODO:
# Verify for two qubit gates. - (SVD and then apply it) (apply and then SVD)
# Evolve half od circuit in mps and half in mpo.
# Evaluate Bell state, epectation value of X at a particular index
# Evaluate X, Y, Z observable
# Check for one/two/three qubits (cover edge cases) for both mps/mpo
test_mpo_construction_from_gate_function()
