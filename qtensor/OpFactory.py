import cirq
import qtree
from functools import partial
# Qiskit >=0.19
#import qiskit.circuit.library as qiskit_lib
#qiskit_lib = qtensor.tools.LasyModule('qiskit.extensions.standard')
from qtensor.tools.lazy_import import qiskit
from qtensor.tools.lazy_import import qiskit_lib
from qtensor.tools.lazy_import import torch
import numpy as np

class OpFactory:
    pass

class TorchFactory:
    pass

class CirqFactory:
    H=cirq.H
    cX=cirq.CX
    Z=cirq.Z
    X=cirq.X

    @staticmethod
    def ZPhase(x, alpha):
        return cirq.ZPowGate(exponent=float(alpha)).on(x)

    @staticmethod
    def YPhase(x, alpha):
        return cirq.YPowGate(exponent=float(alpha)).on(x)

    @staticmethod
    def XPhase(x, alpha):
        return cirq.XPowGate(exponent=float(alpha)).on(x)

    cZ=cirq.CZ

QtreeFactory = qtree.operators
class ZZFull(qtree.operators.ParametricGate):
    name = 'ZZ'
    _changes_qubits=(0,1)
    def gen_tensor(self):
        alpha = self.parameters['alpha']
        p = np.exp(1j*np.pi*alpha/2)
        m = np.exp(-1j*np.pi*alpha/2)
        tensor = np.diag([m, p ,p, m])
        return tensor.reshape((2,)*4)

QtreeFullFactory = qtree.operators_full_matrix
QtreeFullFactory.ZZ = ZZFull

class ZZ(qtree.operators.ParametricGate):
    name = 'ZZ'
    _changes_qubits=tuple()
    parameter_count=1
    def gen_tensor(self):
        alpha = self.parameters['alpha']
        p = np.exp(1j*np.pi*alpha/2)
        m = np.exp(-1j*np.pi*alpha/2)
        tensor = np.array([
             [m, p]
            ,[p, m]
        ])
        return tensor

QtreeFactory.ZZ = ZZ

def torch_j_exp(z):
    """
    https://discuss.pytorch.org/t/complex-functions-exp-does-not-support-automatic-differentiation-for-outputs-with-complex-dtype/98039/3
    """

    z = -1j*z # this is a workarond to fix torch complaining
    # on different types of gradient vs inputs.
    # Just make input complex
    return torch.cos(z) + 1j * torch.sin(z)

class CXTorch(qtree.operators.Gate):
    name = 'ZZ'
    _changes_qubits=(1, )

    def gen_tensor(self):
        return torch.tensor([[[1., 0.],
                          [0., 1.]],
                         [[0., 1.],
                          [1., 0.]]])

class ZZTorch(qtree.operators.ParametricGate):
    name = 'ZZ'
    _changes_qubits=tuple()
    parameter_count=1
    def gen_tensor(self):
        alpha = self.parameters['alpha']
        tensor = torch.tensor([
             [-1, +1]
            ,[+1, -1]
        ])
        return torch_j_exp(1j*tensor*np.pi*alpha/2)

class ZPhaseTorch(qtree.operators.ParametricGate):
    _changes_qubits = tuple()
    parameter_count = 1

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along Z axis"""
        alpha = parameters['alpha']
        t_ = torch.tensor([0, 1])
        return torch_j_exp(1j*t_*np.pi*alpha)

class YPhaseTorch(qtree.operators.ParametricGate):
    parameter_count = 1
    _changes_qubits = (0, )

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along Y axis"""
        alpha = parameters['alpha']

        c = torch.cos(np.pi * alpha / 2)*torch.tensor([[1,0],[0,1]])
        s = torch.sin(np.pi * alpha / 2)*torch.tensor([[0, -1],[1,0]])
        g = torch_j_exp(1j * np.pi * alpha / 2)

        return g*(c + s)

class XPhaseTorch(qtree.operators.ParametricGate):

    _changes_qubits = (0, )
    parameter_count = 1

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along X axis"""
        alpha = parameters['alpha']

        c = torch.cos(np.pi*alpha/2)*torch.tensor([[1,0],[0,1]])
        s = torch.sin(np.pi*alpha/2)*torch.tensor([[0, -1j], [-1j, 0]])
        g = torch_j_exp(1j*np.pi*alpha/2)

        return g*c + g*s

TorchFactory.ZZ = ZZTorch
TorchFactory.XPhase = XPhaseTorch
TorchFactory.YPhase = YPhaseTorch
TorchFactory.ZPhase = ZPhaseTorch
TorchFactory.H = QtreeFactory.H
TorchFactory.Z = QtreeFactory.Z
TorchFactory.X = QtreeFactory.X
TorchFactory.Y = QtreeFactory.Y
TorchFactory.cZ = QtreeFactory.cZ
TorchFactory.cX = CXTorch

# this is a bit ugly, but will work for now
qtree.operators.LABEL_TO_GATE_DICT['zz'] = ZZ

class QiskitFactory:
    @property
    def H(cls):
        return qiskit_lib.HGate

    @property
    def cX(cls):
        # Different versions of qiskit have different names
        try:
            return qiskit_lib.CnotGate
        except:
            return qiskit_lib.CXGate

    @property
    def cZ(cls):
        return qiskit_lib.CZGate

    @property
    def Z(cls):
        return qiskit_lib.ZGate
    
    @staticmethod
    def ZPhase(alpha):
        return qiskit_lib.RZGate(phi=alpha*np.pi)

    @staticmethod
    def YPhase(alpha):
        return qiskit_lib.RYGate(theta=alpha*np.pi)

    @staticmethod
    def XPhase(alpha):
        return qiskit_lib.RXGate(theta=alpha*np.pi)

class CircuitBuilder:
    """ ABC for creating a circuit."""
    operators = OpFactory

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.reset()
        self.qubits = self.get_qubits()

    def get_qubits(self):
        raise NotImplementedError

    def reset(self):
        """ Initialize new circuit """
        raise NotImplementedError

    def inverse(self):
        if not hasattr(self, '_warned'):
            #print('Warning: conjugate is not implemented. Returning same circuit, in case you only care about circuit structure')
            self._warned = True
        return self._circuit

    def apply_gate(self, gate, *qubits, **params):
        self._circuit.append(gate(**params), *qubits)

    @property
    def circuit(self):
        return self._circuit
    @circuit.setter
    def circuit(self, circuit):
        self._circuit = circuit

    def view(self):
        other = type(self)(self.n_qubits)
        other.circuit = self.circuit
        return other

    def copy(self):
        other = type(self)(self.n_qubits)
        other.circuit = self.circuit.copy()
        return other


class CirqBuilder(CircuitBuilder):
    operators = CirqFactory

    def get_qubits(self):
        return [cirq.LineQubit(i) for i in range(self.n_qubits)]

    def reset(self):
        self._circuit = cirq.Circuit()

    def apply_gate(self, gate, *qubits, **params):
        self._circuit.append(gate(*qubits, **params))

    def inverse(self):
        self._circuit = cirq.inverse(self._circuit)

class QtreeBuilder(CircuitBuilder):
    operators = QtreeFactory

    def get_qubits(self):
        return list(range(self.n_qubits))

    def reset(self):
        self._circuit = []

    def apply_gate(self, gate, *qubits, **params):
        self._circuit.append(gate(*qubits, **params))

    def inverse(self):
        # --
        # this in-place creature is to be sure that
        # a new gate is creaded on inverse
        def dagger_gate(gate):
            if hasattr(gate, '_parameters'):
                params = gate._parameters
            else:
                params = {}
            new = type(gate)(*gate._qubits, **params)
            new.name = gate.name + '+'
            new.gen_tensor = partial(gate.dag_tensor, gate)
            return new
        #--

        self._circuit = list(reversed([dagger_gate(g) for g in self._circuit]))

class QiskitBuilder(CircuitBuilder):
    operators = QiskitFactory()

    def get_qubits(self):
        # The ``reset`` should be called first
        return self._circuit.qubits

    def reset(self):
        qreg_size = self.n_qubits
        creg_size = qreg_size
        self._circuit = qiskit.QuantumCircuit(qreg_size, creg_size)

    def apply_gate(self, gate, *qubits, **params):
        self._circuit.append(gate(**params), qubits)

    def inverse(self):
        self._circuit = self._circuit.inverse()

class QtreeFullBuilder(QtreeBuilder):
    operators = QtreeFullFactory

class TorchBuilder(QtreeBuilder):
    operators = TorchFactory
