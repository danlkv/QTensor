from qtensor.simplify_circuit import simplify_circuit
from qtensor.simplify_circuit.gates import zzphase, xphase, yphase, zphase, hadamard, ident, cnot, toffoli, z, cz, x, y

from qtree.operators import XPhase, YPhase, ZPhase, H, cX, Z, cZ, X, Y
from qtree.operators import ParametricGate, Gate

from qtensor.OpFactory import ZZ
from qtensor.OpFactory import CircuitBuilder
from qtensor import ZZQAOAComposer



class SimpFactory:
    ZZ = zzphase
    XPhase = xphase
    YPhase = yphase
    ZPhase = zphase
    H = hadamard
    cX = cnot

class SimpBuilder(CircuitBuilder):
    operators = SimpFactory

    def get_qubits(self):
        return list(range(self.n_qubits))

    def reset(self):
        self._circuit = []

    def apply_gate(self, gate, *qubits, **params):
        gate = gate(*qubits, angle=params.get('alpha',0))
        self._circuit.append(gate)

    def inverse(self):
        def inverse(gate):
            gate.angle *= -1
        self._circuit = list(reversed([inverse(g) for g in self._circuit]))


class SimpQAOAComposer(ZZQAOAComposer):
    """
    Composer that produces circuit form gates
    that are acceptable by simplify_circuit
    functions.

    """
    def _get_builder_class(self):
        return SimpBuilder


def get_simplifiable_circuit_composer(N, p, d):
    import networkx as nx
    import numpy as np
    G = nx.random_regular_graph(d, N)
    gamma, beta = [np.pi/3]*p, [np.pi/2]*p

    comp = SimpQAOAComposer(G, gamma=gamma, beta=beta)
    return comp




GATE_MAP = {
    zzphase:  ZZ,
    xphase: XPhase,
    yphase: YPhase,
    zphase: ZPhase,
    hadamard: H,
    cnot: cX,
    cz: cZ,
    z: Z,
    x: X,
    y: Y,
}


def simplify_qtree_circuit(qtreeCircuit):
    """
    Simplify circuit by using commutative relations on pairs of gates

    Args:
        qtreeCircuit (list[qtree.Gate]): circuit to simplify

    Returns list[qtree.Gate]: simplified circuit

    """
    circuit = []
    for qtreeGate in qtreeCircuit:
        inv_map = {v:k for k, v in GATE_MAP.items()}
        GateClass = inv_map[qtreeGate.__class__]
        try:
            sign_of_param = -1 if '+' in qtreeGate.name else 1
            gate = GateClass(*qtreeGate.qubits, angle=sign_of_param*qtreeGate.parameters['alpha'])
        except KeyError:
            gate = GateClass(*qtreeGate.qubits)
        circuit.append(gate)

    simplified = simplify_circuit(circuit)

    qtree_circuit = []
    for gate in simplified:
        GateClass = GATE_MAP[gate.__class__]
        if issubclass(GateClass, ParametricGate):
            qtree_gate = GateClass(*gate.index, alpha=gate.angle)
        else:
            qtree_gate = GateClass(*gate.index)
        qtree_circuit.append(qtree_gate)

    return qtree_circuit




