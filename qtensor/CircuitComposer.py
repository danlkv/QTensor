from loguru import logger as log
import networkx as nx
import qtensor
from qtensor import utils
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

    def conjugate(self):
        # changes builder.circuit, hence self.circuit()
        self.builder.conjugate()
    #--

    def layer_of_Hadamards(self):
        for q in self.qubits:
            self.apply_gate(self.operators.H, q)

    def expectation(self, operator, *qubits, **params):
        """
        Args:
            operator: an element from OpFactory
            qubits: qubits to apply the gate to
            params: params to pass on application gate

        Returns:
            a circuit, 0th amplitude of which evaluates to expectation value
        """

        # TODO: should return a tensor network,
        # no circuit returns expectations
        first_part = self.builder.copy()
        first_part.apply_gate(operator, *qubits, **params)
        second_part = self.builder.copy()
        second_part.inverse()

        circ = first_part.circuit + second_part.circuit
        do_simplify = len(circ) < 100
        if do_simplify:
            try:
                circ = qtensor.simplify_circuit.simplify_qtree_circuit(circ)
            except Exception as e:

                pass
                #print('failed to simplify:', type(e), e)

        return circ


class OldQAOAComposer(CircuitComposer):
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
        # Will need to deprecate stateful API and return the circuit
        self.ansatz_state()
        self.energy_edge(i, j)
        first_part = self.builder.circuit

        self.builder.reset()
        self.ansatz_state()
        self.builder.inverse()
        second_part = self.builder.circuit

        self.circuit = first_part + second_part

    def energy_expectation_lightcone(self, edge):
        G = self.graph
        gamma, beta = self.params['gamma'], self.params['beta']
        i,j = edge
        graph = utils.get_edge_subgraph(G, edge, len(gamma))
        log.debug('Subgraph nodes: {}, edges: {}', graph.number_of_nodes(), graph.number_of_edges())
        self.n_qubits = graph.number_of_nodes()
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
    def mixer_operator(self, beta, nodes=None):
        if nodes is None: nodes = self.graph.nodes()
        for n in nodes:
            qubit = self.qubits[n]
            self.x_term(qubit, beta)

    def append_zz_term(self, q1, q2, gamma):
        self.apply_gate(self.operators.cX, q1, q2)
        self.apply_gate(self.operators.ZPhase, q2, alpha=2*gamma)
        self.apply_gate(self.operators.cX, q1, q2)
    def cost_operator_circuit(self, gamma, edges=None):
        if edges is None: edges = self.graph.edges()
        for i, j in edges:
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

    def energy_edge(self, i, j):
        u, v = self.qubits[i], self.qubits[j]
        self.apply_gate(self.operators.Z, u)
        self.apply_gate(self.operators.Z, v)


class QAOAComposer(OldQAOAComposer):
    def cone_ansatz(self, edge):
        beta, gamma = self.params['beta'], self.params['gamma']

        assert(len(beta) == len(gamma))
        p = len(beta) # infering number of QAOA steps from the parameters passed
        self.layer_of_Hadamards()
        # second, apply p alternating operators
        cone_base = self.graph

        for i, g, b in zip(range(p, 0, -1), gamma, beta):
            self.graph = utils.get_edge_subgraph(cone_base, edge, i)
            self.cost_operator_circuit(g)
            self.graph = utils.get_edge_subgraph(cone_base, edge, i-1)
            self.mixer_operator(b)
        self.graph = cone_base


    def energy_expectation(self, i, j):
        # Will need to deprecate stateful API and return the circuit
        self.cone_ansatz(edge=(i, j))
        self.energy_edge(i, j)
        first_part = self.builder.circuit
        self.builder.reset()

        self.cone_ansatz(edge=(i, j))
        self.builder.inverse()
        second_part = self.builder.circuit

        self.circuit = first_part + second_part


class ZZQAOAComposer(QAOAComposer):
    def append_zz_term(self, q1, q2, gamma):
        self.apply_gate(self.operators.ZZ, q1, q2, alpha=2*gamma)

class QAOAComposerChords(ZZQAOAComposer):
    def cone_ansatz(self, edge):
        beta, gamma = self.params['beta'], self.params['gamma']

        assert(len(beta) == len(gamma))
        p = len(beta) # infering number of QAOA steps from the parameters passed
        self.layer_of_Hadamards()
        # second, apply p alternating operators
        cone_base = self.graph

        for i, g, b in zip(range(p, 0, -1), gamma, beta):
            self.graph = utils.get_edge_subgraph_old(cone_base, edge, i)
            self.cost_operator_circuit(g)
            self.mixer_operator(b)
        self.graph = cone_base


class WeightedZZQAOAComposer(ZZQAOAComposer):

    def cost_operator_circuit(self, gamma, edges=None):
        for i, j, w in self.graph.edges.data('weight', default=1):
            u, v = self.qubits[i], self.qubits[j]
            self.append_zz_term(u, v, gamma*w)
