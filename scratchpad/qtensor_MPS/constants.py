import numpy as np

pauli_matrices = ["X", "Z", "I", "Y"]
hmatrix = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex64)
imatrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex64)
xmatrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
ymatrix = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex64)
zmatrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

_cnot_matrix = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)
cnot_matrix = np.reshape(_cnot_matrix, newshape=(2, 2, 2, 2))

_swap_matrix = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
swap_matrix = np.reshape(_swap_matrix, newshape=(2, 2, 2, 2))

_hbar = 1.0545718 * 10e-34
_sigma_z = _hbar * 0.5 * zmatrix
_sigma_x_pos = _hbar * 0.5 * (xmatrix + 1j * ymatrix)
_sigma_x_neg = _hbar * 0.5 * (xmatrix - 1j * ymatrix)

_sigma_z_sigma_z_gate_matrix = np.kron(_sigma_z, _sigma_z)
sigma_z_sigma_z_gate_matrix = _sigma_z_sigma_z_gate_matrix

# TODO: Check zz matrix is diagonal
# TODO: What does it look if hamiltonian is e^(I) / e^(0)

_sigma_x_pos_sigma_x_neg_gate_matrix = np.kron(_sigma_x_pos, _sigma_x_neg)
sigma_x_pos_sigma_x_neg_gate_matrix = np.reshape(
    _sigma_x_pos_sigma_x_neg_gate_matrix, newshape=(2, 2, 2, 2)
)

_sigma_x_neg_sigma_x_pos_gate_matrix = np.kron(_sigma_x_neg, _sigma_x_pos)
sigma_x_neg_sigma_x_pos_gate_matrix = np.reshape(
    _sigma_x_neg_sigma_x_pos_gate_matrix, newshape=(2, 2, 2, 2)
)
# _ising_hamiltonian_matrix = sigma_z_sigma_z_gate_matrix + 0.5 * sigma_x_pos_sigma_x_neg_gate_matrix + 0.5 * sigma_x_neg_sigma_x_pos_gate_matrix
_ising_hamiltonian_matrix = sigma_z_sigma_z_gate_matrix
ising_hamiltonian_matrix = np.reshape(_ising_hamiltonian_matrix, newshape=(2, 2, 2, 2))
