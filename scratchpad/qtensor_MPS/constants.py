import numpy as np

_hmatrix = (
    1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex64)
)
_imatrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex64)
_xmatrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
_ymatrix = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex64)
_zmatrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)

_cnot_matrix = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)
_cnot_matrix = np.reshape(_cnot_matrix, newshape=(2, 2, 2, 2))

_swap_matrix = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
_swap_matrix = np.reshape(_swap_matrix, newshape=(2, 2, 2, 2))