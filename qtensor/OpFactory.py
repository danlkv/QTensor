import cirq
import qtree
import qiskit
# Qiskit >=0.19
#import qiskit.circuit.library as qiskit_lib

import qiskit.extensions.standard as qiskit_lib
import numpy as np

class OpFactory:
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
    def XPhase(x, alpha):
        return cirq.XPowGate(exponent=float(alpha)).on(x)

    cZ=cirq.CZ

QtreeFactory = qtree.operators

class CC(qtree.operators.ParametricGate):
    name = 'CC'
    _changes_qubits=tuple()
    def gen_tensor(self):
        alpha = self.parameters['alpha']
        ep = np.exp(1j*np.pi*alpha/2)
        em = np.exp(-1j*np.pi*alpha/2)
        tensor = np.array([
            [ep,em]
            ,[em,ep]
        ])
        return tensor


#QtreeFactory.CC = CC

class QiskitFactory:
    H=qiskit_lib.HGate
    cX=qiskit_lib.CnotGate

    @staticmethod
    def ZPhase(alpha):
        return qiskit_lib.RZGate(phi=alpha*np.pi)

    @staticmethod
    def XPhase(alpha):
        return qiskit_lib.RXGate(theta=alpha*np.pi)

    cZ=qiskit_lib.CzGate
    Z=qiskit_lib.ZGate

class CircuitBuilder:
    """ ABC for creating a circuit."""
    operators = OpFactory

    def __init__(self, n_qubits, **params):
        self.n_qubits = n_qubits
        self.reset()
        self.qubits = self.get_qubits()

    def get_qubits(self):
        raise NotImplementedError

    def reset(self):
        """ Initialize new circuit """
        raise NotImplementedError

    def conjugate(self):
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

    def inverse(self):
        """ Reverse order and conjugate gates. """
        raise NotImplementedError


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
        self._circuit = list(reversed([g.dagger_me() for g in self._circuit]))

class QiskitBuilder(CircuitBuilder):
    operators = QiskitFactory

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
