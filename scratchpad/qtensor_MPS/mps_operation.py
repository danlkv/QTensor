import numpy as np
import tensornetwork as tn
import numpy as np
from constants import _xmatrix, _cnot_matrix, _hmatrix
from copy import deepcopy

def xgate() -> tn.Node:
    return tn.Node(deepcopy(_xmatrix), name="xgate")

def cnot() -> tn.Node:
    return tn.Node(deepcopy(_cnot_matrix), name="cnot")

def hgate() -> tn.Node:
    return tn.Node(deepcopy(_hmatrix), name="hgate")

