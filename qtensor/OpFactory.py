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

class CircuitBuilder:
    """ ABC for creating a circuit."""
    operators = OpFactory

    def __init__(self, n_qubits, **params):
        self.n_qubits = n_qubits
        self.circuit = self.get_circuit()
        self.qubits = self.get_qubits()

    def get_qubits(self):
        raise NotImplementedError

    def get_circuit(self):
        raise NotImplementedError

    def conjugate(self):
        if not hasattr(self, '_warned'):
            #print('Warning: conjugate is not implemented. Returning same circuit, in case you only care about circuit structure')
            self._warned = True
        return self.circuit

    def apply_gate(self, gate, *qubits, **params):
        self.circuit.append(gate(**params), *qubits)


class CirqBuilder(CircuitBuilder):
    operators = CirqFactory

    def get_qubits(self):
        return [cirq.LineQubit(i) for i in range(self.n_qubits)]
    def get_circuit(self):
        return cirq.Circuit()

    def apply_gate(self, gate, *qubits, **params):
        self.circuit.append(gate(*qubits, **params))

class QtreeBuilder(CircuitBuilder):
    operators = QtreeFactory

    def get_qubits(self):
        return list(range(self.n_qubits))
    def get_circuit(self):
        return []

    def apply_gate(self, gate, *qubits, **params):
        self.circuit.append(gate(*qubits, **params))

    def conjugate(self):
        self.circuit = [g.dagger_me() for g in self.circuit]
        return self.circuit

class QiskitBuilder(CircuitBuilder):
    operators = QiskitFactory

    def get_qubits(self):
        # The ``get_circuit`` should be called first
        return self.circuit.qubits

    def get_circuit(self):
        qreg_size = self.n_qubits
        creg_size = qreg_size
        return qiskit.QuantumCircuit(qreg_size, creg_size)

    def apply_gate(self, gate, *qubits, **params):

        self.circuit.append(gate(**params), qubits)
