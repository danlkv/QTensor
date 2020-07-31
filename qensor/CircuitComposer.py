from .OpFactory import CircuitCreator

class CircuitComposer(CircuitCreator):
    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        self.params = params
        self.qubits = self.get_qubits()
        self.circuit = self.get_circuit()

    def layer_of_Hadamards(self):
        for q in self.qubits:
            self.circuit.append(self.operators.H(q))

    def create(self):
        raise NotImplementedError


class QAOAComposer(CircuitComposer):
    def __init__(self, graph, *args, **kwargs):
        n_qubits = graph.number_of_nodes()
        super().__init__(n_qubits, *args, **kwargs)

        self.graph = graph

    def x_term(self, u, beta):
        #self.circuit.append(self.operators.H(u))
        self.circuit.append(self.operators.XPhase(u, alpha=2*beta))
        #self.circuit.append(self.operators.H(u))
    def mixer_operator(self, beta):
        G = self.graph
        for n in G:
            qubit = self.qubits[n]
            self.x_term(qubit, beta)

    def append_zz_term(self, q1, q2, gamma):
        try:
            self.circuit.append(self.operators.CC(q1, q2, alpha=2*gamma))
        except AttributeError:
            pass
        self.circuit.append(self.operators.cX(q1, q2))
        self.circuit.append(self.operators.ZPhase(q2, alpha=2*gamma))
        self.circuit.append(self.operators.cX(q1, q2))
    def cost_operator_circuit(self, gamma):
        for i, j in self.graph.edges():
            u, v = self.qubits[i], self.qubits[j]
            self.append_zz_term(u, v, gamma)


    def ansatz_state(self, operators="diagonal"):
        beta, gamma = self.params['beta'], self.params['gamma']

        assert(len(beta) == len(gamma))
        p = len(beta) # infering number of QAOA steps from the parameters passed
        self.layer_of_Hadamards()
        # second, apply p alternating operators
        for i in range(p):
            self.cost_operator_circuit(gamma[i])
            self.mixer_operator(beta[i])
        return self.circuit

    def energy_edge(self, i, j):
        #self.circuit.append(self.operators.CC(u, v))
        u, v = self.qubits[i], self.qubits[j]
        self.circuit.append(self.operators.Z(u))
        self.circuit.append(self.operators.Z(v))

