import numpy as np
import tensornetwork as tn
import numpy as np
from constants import *
from copy import deepcopy

def xgate() -> tn.Node:
    return tn.Node(deepcopy(xmatrix), name="xgate")

def zgate() -> tn.Node:
    return tn.Node(deepcopy(zmatrix), name="xgate")

def cnot() -> tn.Node:
    return tn.Node(deepcopy(cnot_matrix), name="cnot")

def hgate() -> tn.Node:
    return tn.Node(deepcopy(hmatrix), name="hgate")

def sigmaZZ(t) -> tn.Node:
    return tn.Node(deepcopy(np.exp(sigma_z_sigma_z_gate_matrix * -1j * t * 0.5)), name="sigmaZZ")

def sigmaXposXneg(t) -> tn.Node:
    return tn.Node(deepcopy(np.exp(sigma_x_pos_sigma_x_neg_gate_matrix * -1j * t * 0.5)), name="sigmaXposXneg")

def sigmaXnegXpos(t) -> tn.Node:
    return tn.Node(deepcopy(np.exp(sigma_x_neg_sigma_x_pos_gate_matrix * -1j * t * 0.5 )), name="sigmaXposXneg")

def isingHamiltonian(t)-> tn.Node:
    return tn.Node(deepcopy(np.exp(ising_hamiltonian_matrix * -1j * t * 0.5)))

