import cirq
import qtree

class OpFactory:
    pass


class CirqFactory:
    H=cirq.H
    cX=cirq.CX

    @staticmethod
    def ZPhase(x, alpha):
        return cirq.ZPowGate(exponent=float(alpha)).on(x)

    cZ=cirq.CZ

QtreeFactory = qtree.operators


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


class CirqCreator(CircuitCreator):
    operators = CirqFactory

    def get_qubits(self):
        return [cirq.LineQubit(i) for i in range(self.n_qubits)]
    def get_circuit(self):
        return cirq.Circuit()

class QtreeCreator(CircuitCreator):
    operators = QtreeFactory

    def get_qubits(self):
        return list(range(self.n_qubits))
    def get_circuit(self):
        return []
