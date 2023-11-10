import numpy as np
import tensornetwork as tn
import numpy as np
from constants import *
from copy import deepcopy
from scipy.linalg import qr

def xgate() -> tn.Node:
    return tn.Node(deepcopy(xmatrix), name="xgate")

def igate() -> tn.Node:
    return tn.Node(deepcopy(imatrix), name="igate")

def zgate() -> tn.Node:
    return tn.Node(deepcopy(zmatrix), name="xgate")

def cnot() -> tn.Node:
    return tn.Node(deepcopy(cnot_matrix), name="cnot")

def hgate() -> tn.Node:
    return tn.Node(deepcopy(hmatrix), name="hgate")


def sigmaRZZ(t) -> tn.Node:
    _hbar = 1.0545718*10e-34
    theta = 1j * t * 0.5 * _hbar * 0.5

    gate_matrix = np.array(
        [
            [np.exp(theta/2), 0.0, 0.0, 0.0],
            [0.0, np.exp(theta/2), 0.0, 0.0],
            [0.0, 0.0, np.exp(theta/2), 0.0],
            [0.0, 0.0, 0.0, np.exp(theta/2)],
        ]
    )
    tensor = np.reshape(gate_matrix, newshape=(2,2,2,2))
    return tn.Node(deepcopy(tensor), name="sigmaZZ")

def sigmaZZ(t) -> tn.Node:
    gate_matrix = np.exp(sigma_z_sigma_z_gate_matrix * -1j * t * 0.5)
    gate_matrix = gate_matrix
    q, r = np.linalg.qr(gate_matrix)
    diag = np.diag(r).copy()
    diag /= np.abs(diag)
    tensor = q * diag
    tensor = np.reshape(tensor, newshape=(2,2,2,2))
    return tn.Node(deepcopy(tensor), name="sigmaZZ")

def sigmaXposXneg(t) -> tn.Node:
    return tn.Node(deepcopy(np.exp(sigma_x_pos_sigma_x_neg_gate_matrix * -1j * t * 0.5)), name="sigmaXposXneg")

def sigmaXnegXpos(t) -> tn.Node:
    return tn.Node(deepcopy(np.exp(sigma_x_neg_sigma_x_pos_gate_matrix * -1j * t * 0.5 )), name="sigmaXposXneg")

def isingHamiltonian(t)-> tn.Node:
    return tn.Node(deepcopy(np.exp(ising_hamiltonian_matrix * -1j * t * 0.5)))

