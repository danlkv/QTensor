import cirq
import qtree
#import qiskit.circuit.library as qiskit_lib
import numpy as np

class OpFactory:
    pass


class CirqFactory:
    H=cirq.H
    cX=cirq.CX

    @staticmethod
    def ZPhase(x, alpha):
        return cirq.ZPowGate(exponent=float(alpha)).on(x)

    @staticmethod
    def XPhase(x, alpha):
        return cirq.XPowGate(exponent=float(alpha)).on(x)

    cZ=cirq.CZ

QtreeFactory = qtree.operators

class CC(qtree.operators.Gate):
    name = 'CC'
    _changes_qubits=tuple()
    def gen_tensor(self):
        tensor = np.array([
            [0,1]
            ,[1,1]
        ])
        return tensor


QtreeFactory.CC = CC

"""
class QiskitFactory:
    H=qiskit_lib.HGate
    cX=qiskit_lib.CXGate

    @staticmethod
    def ZPhase(x, alpha):
        return qiskit_lib.RZGate(phi=alpha*np.pi)

    cZ=qiskit_lib.CZGate

"""
class CircuitCreator:
    operators = OpFactory

    def __init__(self, n_qubits, **params):
        self.n_qubits = n_qubits
        self.qubits = self.get_qubits()
        self.circuit = self.get_circuit()

    def get_qubits(self):
        raise NotImplementedError

    def get_circuit(self):
        raise NotImplementedError

    def apply_gate(self, gate, *qubits, **params):
        self.circuit.append(gate(**params), *qubits)


class CirqCreator(CircuitCreator):
    operators = CirqFactory

    def get_qubits(self):
        return [cirq.LineQubit(i) for i in range(self.n_qubits)]
    def get_circuit(self):
        return cirq.Circuit()

    def apply_gate(self, gate, *qubits, **params):
        self.circuit.append(gate(*qubits, **params))

class QtreeCreator(CircuitCreator):
    operators = QtreeFactory

    def get_qubits(self):
        return list(range(self.n_qubits))
    def get_circuit(self):
        return []

    def apply_gate(self, gate, *qubits, **params):
        self.circuit.append(gate(*qubits, **params))

