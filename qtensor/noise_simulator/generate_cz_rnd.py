import numpy as np
import qtree
import qtensor

def register_qiskit_gate():
    import qiskit
    qtensor.OpFactory.QiskitFactory.PauliGate = lambda x, alpha, beta, gamma: qiskit.extensions.UnitaryGate(
        qtensor.tools.unitary.random_pauli(alpha, beta, gamma),
        label='RandomPauliGate'
    )

def pauli_unitary(a, b, c):
    """
    Returns exp(-i*c*pauli.m)
    where m = (sin(a)cos(b), sin(a)sin(b), cos(b))
    """
    v = np.array([
        np.sin(a)*np.cos(b),
        np.sin(a)*np.sin(b),
        np.cos(a)
    ])
    paulis = np.array([
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, 1j], [-1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ])
    R = np.sum([paulis[i]*v[i] for i in range(3)], axis=0)
    return np.cos(c)*np.eye(2) - 1j*np.sin(c)*R

class PauliGate(qtree.operators.ParametricGate):
    name = "Pauli"
    def __init__(self, *args, **kwargs):
        self.ar = pauli_unitary(kwargs['alpha'], kwargs['beta'], kwargs['gamma'])
        super().__init__(*args, **kwargs)
        # used by data_key in global storage

    def gen_tensor(self):
        return self.ar

    _changes_qubits = (0,)


def get_cx_random_circuit(N, d, angle_sequence=None):
    if angle_sequence is None:
        angle_sequence = np.random.uniform(0, 2*np.pi, size=(N*d*2, 3))
    circuit = []
    k = 0
    for l in range(d*2):
        layer = []
        for i in range(N):
            gate = PauliGate(
                i,
                alpha=angle_sequence[k, 0],
                beta=angle_sequence[k, 1],
                gamma=angle_sequence[k, 2]
            )
            layer.append(gate)
            k += 1
        s = l % 2
        #for s in range(2):
        for i in range(N//2-s):
            u, v = (s+2*i, s+2*i+1)
            g = qtree.operators.cZ(u, v)
            layer.append(g)
        circuit.append(layer)
    return circuit


class CZBrickworkComposer(qtensor.CircuitComposer):
    def _get_builder(self):
        return qtensor.QtreeBuilder(self.n_qubits)

    def __init__(self, S, *args, **kwargs):
        self.S = S
        self.n_qubits = S*S
        super().__init__(self.n_qubits, *args, **kwargs)

    def apply_layer(self, mode):
        switch = 'n' in mode
        start = lambda s: 0 if s else 1
        end_i = self.S if 'v' in mode else (self.S - 1)
        end_j = self.S if 'h' in mode else self.S - 1
        for j in range(0, end_j, 1):
            for i in range(start(switch), end_i, 2):
                A = i+self.S*j
                if 'v' in mode:
                    # vertical
                    B = i + self.S*(j+1)
                else:
                    # horisontal
                    B = i + 1 + self.S*j
                self.two_qubit(A, B)
            switch = not switch

    def two_qubit(self, u, v):
        self.apply_gate(self.builder.operators.cZ, u, v)

    def one_qubit(self, u, params):
        options = [self.builder.operators.X, self.builder.operators.Y, self.builder.operators.Z]
        gate = np.random.choice(options)
        self.apply_gate(gate, u)

    def one_qubit_layer(self, params=None):
        if params is None:
            params = 2*np.pi*np.random.rand(self.n_qubits, 3)
        for i in range(self.n_qubits):
            self.one_qubit(i, params[i])

    def two_qubit_rnd(self, layers=10):
        modes = ['vp', 'vn', 'hp', 'hn']
        for i in range(layers):
            for mode in modes:
                self.one_qubit_layer()
                self.apply_layer(mode)


def get_cz_circ(S, d):
    comp = CZBrickworkComposer(S)
    comp.two_qubit_rnd(layers=d)
    return comp.circuit

class QiskitCZBrickworkComposer(CZBrickworkComposer):
    def _get_builder(self):
        return qtensor.QiskitBuilder(self.n_qubits)


if __name__ == "__main__":
    circ = get_cz_circ(3, 5)
    print(circ)
