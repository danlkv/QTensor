from loguru import logger as log
from qtensor.utils import get_edge_subgraph
import networkx as nx
from .OpFactory import CircuitBuilder

class CircuitComposer():
    """ Director for CircuitBuilder, but with a special way to get the builder"""
    Bulider = CircuitBuilder
    def __init__(self, *args, **params):
        self.params = params
        self.builder = self._get_builder()
        self.n_qubits = self.builder.n_qubits

    #-- Setting up the builder
    def _get_builder_class(self):
        raise NotImplementedError

    def _get_builder(self):
        return self._get_builder_class()()


    #-- Mocking some of bulider behaviour
    @property
    def operators(self):
        return self.builder.operators

    @property
    def circuit(self):
        return self.builder.circuit
    @circuit.setter
    def circuit(self, circuit):
        self.builder.circuit = circuit

    @property
    def qubits(self):
        return self.builder.qubits
    @qubits.setter
    def qubits(self, qubits):
        self.builder.qubits = qubits

    def apply_gate(self, gate, *qubits, **params):
        self.builder.apply_gate(gate, *qubits, **params)
    #--

    def layer_of_Hadamards(self):
        for q in self.qubits:
            self.apply_gate(self.operators.H, q)


class QAOAComposer(CircuitComposer):
    """ Abstract base class for QAOA Director """
    def __init__(self, graph, *args, **kwargs):
        self.n_qubits = graph.number_of_nodes()
        super().__init__(*args, **kwargs)

        self.graph = graph

    def _get_builder(self):
        return self._get_builder_class()(self.n_qubits)

    @classmethod
    def _get_of_my_type(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def energy_expectation(self, i, j):
        G = self.graph
        self.ansatz_state()
        self.energy_edge(i, j)

        beta, gamma = self.params['beta'], self.params['gamma']
        conjugate = self._get_of_my_type(G, beta=beta, gamma=gamma)
        conjugate.ansatz_state()
        conjugate.builder.conjugate()

        self.circuit = self.circuit + list(reversed(conjugate.circuit ))
        return self.circuit

    def energy_expectation_lightcone(self, edge):
        G = self.graph
        gamma, beta = self.params['gamma'], self.params['beta']
        i,j = edge
        # TODO: take only a neighbourhood part of the graph
        graph = get_edge_subgraph(G, edge, len(gamma))
        log.debug('Subgraph nodes: {}, edges: {}', graph.number_of_nodes(), graph.number_of_edges())
        graph = get_edge_subgraph(G, edge, len(gamma))
        mapping = {v:i for i, v in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping, copy=True)

        i,j = mapping[i], mapping[j]
        composer = self._get_of_my_type(graph, beta=beta, gamma=gamma)
        composer.energy_expectation(i,j)
        self.circuit = composer.circuit
        # return composer


    def x_term(self, u, beta):
        #self.circuit.append(self.operators.H(u))
        self.apply_gate(self.operators.XPhase, u, alpha=2*beta)
        #self.circuit.append(self.operators.H(u))
    def mixer_operator(self, beta):
        G = self.graph
        for n in G:
            qubit = self.qubits[n]
            self.x_term(qubit, beta)

    def append_zz_term(self, q1, q2, gamma):
        try:
            self.apply_gate(self.operators.CC, q1, q2, alpha=2*gamma)
        except AttributeError:
            pass
        self.apply_gate(self.operators.cX, q1, q2)
        self.apply_gate(self.operators.ZPhase, q2, alpha=2*gamma)
        self.apply_gate(self.operators.cX, q1, q2)
    def cost_operator_circuit(self, gamma):
        for i, j in self.graph.edges():
            u, v = self.qubits[i], self.qubits[j]
            self.append_zz_term(u, v, gamma)


    def ansatz_state(self):
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
        self.apply_gate(self.operators.Z, u)
        self.apply_gate(self.operators.Z, v)

