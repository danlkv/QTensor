import numpy as np
import networkx as nx

from qtensor.CircuitComposer import QAOAComposer
from qtensor.Simulate import QtreeSimulator
from qtensor.Simulate import CirqSimulator
from qtensor.QAOASimulator import QAOASimulator
from qtensor.OpFactory import CirqBuilder
from qtensor.OpFactory import QiskitBuilder
from qtensor.OpFactory import QtreeBuilder
from qtensor import utils
from loguru import logger as log

from docplex.mp.model import Model

# Appropriate locations:
#   utils.py:
#       graph_to_docplexqubo
#       graph_to_qiskitqubo
#       docplexqubo_to_graph
#       qiskitqubo_to_graph
#       get_node_subgraph
#   CircuitComposer.py:
#       IsingQAOAComposer
#       IsingZZQAOAComposer
#   QAOASimulator.py:
#       IsingQAOASimulator
#       QtreeIsingQAOASimulator
#       CirqIsingQAOASimulator
#   __init__.py:
#       CirqIsingQAOAComposer
#       QiskitIsingQAOAComposer
#       QtreeIsingQAOAComposer
#       CirqIsingZZQAOAComposer
#       QiskitIsingZZQAOAComposer
#       QtreeIsingZZQAOAComposer

def graph_to_docplexqubo(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    
    mdl = Model()
    x = mdl.binary_var_list(f'x{i}' for i in range(n))
    
    node_var_dict = {nodes[i]: var for (i, var) in enumerate(x)}
    
    # Objective construction ############
    if 'offset' in G.graph:
        objective = G.graph['offset']
    else:
        objective = 0
    
    for i, w in G.nodes.data('weight'):
        if w is not None:
            objective += -2*w*node_var_dict[i] + w
    
    for i, j, w in G.edges.data('weight'):
        if w is not None:
            objective += 4*w*node_var_dict[i]*node_var_dict[j]
            objective -= 2*w*(node_var_dict[i]+node_var_dict[j])
            objective += w
    #####################################
    
    mdl.minimize(objective)
    return mdl

def graph_to_qiskitqubo(G):
    try:
        # needs qiskit_optimization package (qiskit[optimization])
        from qiskit_optimization import QuadraticProgram
    except ModuleNotFoundError:
        try:
            # deprecated
            from qiskit.optimization import QuadraticProgram
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Needs qiskit_optimization module to work.")
    
    mdl = graph_to_docplexqubo(G)
    qp = QuadraticProgram()
    qp.from_docplex(mdl)
    return qp

def _qubo_to_graph(n, qubo_constant, qubo_linear, qubo_quadratic):
    graph_constant = qubo_constant
    graph_linear = dict()
    graph_quadratic = dict()
    
    for i in qubo_linear:
        tmp = qubo_linear[i]/2
        graph_constant += tmp
        graph_linear[i] = -tmp
    
    for i, j in qubo_quadratic:
        tmp = qubo_quadratic[(i, j)]/4
        graph_constant += tmp
        
        for k in (i, j):
            if k in qubo_linear:
                graph_linear[k] += -tmp
            else:
                graph_linear[k] += -tmp
        
        if i == j:
            graph_constant += tmp
        else:
            graph_quadratic[(min(i, j), max(i, j))] = tmp
    
    G = nx.Graph(offset=graph_constant)
    G.add_nodes_from(range(n))
    for i in graph_linear:
        G.nodes[i]['weight'] = graph_linear[i]
        # G.add_edge(i, i, weight=graph_linear[i])
    
    for i, j in graph_quadratic:
        G.add_edge(i, j, weight=graph_quadratic[(i,j)])
    
    return G

def docplexqubo_to_graph(mdl): # mdl: DOcplex model
    n = mdl.number_of_binary_variables
    qubo_objective = mdl.get_objective_expr()
    
    var_map = {var: i for (i, var) in enumerate(mdl.iter_binary_vars())}
    
    qubo_constant = qubo_objective.constant
    qubo_linear = {
        var_map[var]: coeff for (var, coeff) in qubo_objective.iter_terms()
    }
    qubo_quadratic = {
        (var_map[var1], var_map[var2]): coeff for (var1, var2, coeff) in qubo_objective.iter_quad_triplets()
    }
    
    return _qubo_to_graph(n, qubo_constant, qubo_linear, qubo_quadratic)

def qiskitqubo_to_graph(qp): # qp: qiskit_optimization quadratic program
    n = qp.get_num_binary_vars()
    qubo_objective = qp.objective
    
    qubo_constant = qubo_objective.constant
    qubo_linear = qubo_objective.linear.to_dict()
    qubo_quadratic = qubo_objective.quadratic.to_dict()
    
    return _qubo_to_graph(n, qubo_constant, qubo_linear, qubo_quadratic)

def get_node_subgraph(G, node, dist): # Approach (b) specific
    nodes_groups = utils.nodes_group_by_distance(G, [node], dist)
    all_nodes = sum(nodes_groups.values(), [])
    subgraph = G.subgraph(all_nodes).copy()
    farthest_nodes = nodes_groups[dist]
    #   for v in farthest_nodes:
    #       u, w = edge
    #       shpu, shpw = nx.shortest_path(G, u, v), nx.shortest_path(G, w, v)
    #       print('shp, dist', len(shpu), len(shpw), dist)
    #       assert (len(shpu) == dist + 1) or (len(shpw) == dist+1)
    edges_to_delete = []
    for u, v in subgraph.edges():
        if (u in farthest_nodes) and (v in farthest_nodes):
            edges_to_delete.append((u,v))
    #print('removing edges', edges_to_delete)
    subgraph.remove_edges_from(edges_to_delete)
    return subgraph

class IsingQAOAComposer(QAOAComposer):
    def node_energy_expectation_lightcone(self, node): # Approach (b) specific
        G = self.graph
        gamma, beta = self.params['gamma'], self.params['beta']
        graph = get_node_subgraph(G, node, len(gamma))
        log.debug('Subgraph nodes: {}, edges: {}', graph.number_of_nodes(), graph.number_of_edges())
        self.n_qubits = graph.number_of_nodes()
        mapping = {v:i for i, v in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping, copy=True)
        
        node = mapping[node]
        composer = self._get_of_my_type(graph, beta=beta, gamma=gamma)
        composer.node_energy_expectation(node)
        self.circuit = composer.circuit
    
    def node_energy_expectation(self, node): # Approach (b) specific
        # Will need to deprecate stateful API and return the circuit
        self.ansatz_state()
        self.energy_node(node)
        first_part = self.builder.circuit

        self.builder.reset()
        self.ansatz_state()
        self.builder.inverse()
        second_part = self.builder.circuit

        self.circuit = first_part + second_part
    
    def energy_node(self, i): # Approach (b) specific
        u = self.qubits[i]
        self.apply_gate(self.operators.Z, u)
    
    # def energy_edge(self, i, j): # Approach (a) specific
    #     u, v = self.qubits[i], self.qubits[j]
    #     self.apply_gate(self.operators.Z, u)
    #     if i != j:
    #         self.apply_gate(self.operators.Z, v)
    
    def cost_operator_circuit(self, gamma, edges=None):
        for i, w in self.graph.nodes.data('weight'):
            if w is not None:
                u = self.qubits[i]
                self.apply_gate(self.operators.ZPhase, u, alpha=2*gamma*w)
        
        for i, j, w in self.graph.edges.data('weight'):
            if w is not None:
                u, v = self.qubits[i], self.qubits[j]
                self.append_zz_term(u, v, gamma*w)
        
        # for i, j, w in self.graph.edges.data('weight', default=1):
        #     u, v = self.qubits[i], self.qubits[j]
        #     if i != j:
        #         self.append_zz_term(u, v, gamma*w)
        #     else:
        #         self.apply_gate(self.operators.ZPhase, u, alpha=2*gamma*w)

class IsingZZQAOAComposer(IsingQAOAComposer):
    def append_zz_term(self, q1, q2, gamma):
        self.apply_gate(self.operators.ZZ, q1, q2, alpha=2*gamma)

class IsingQAOASimulator(QAOASimulator):
    def energy_expectation_parallel(self, *args, **kwargs):
        raise NotImplementedError
    
    def energy_expectation_mpi(self, *args, **kwargs):
        raise NotImplementedError
    
    def _node_energy_circuit(self, G, gamma, beta, node): # Approach (b) specific
        composer = self.composer(G, gamma=gamma, beta=beta)
        composer.node_energy_expectation_lightcone(node)
        return composer.circuit
    
    def energy_expectation(self, G, gamma, beta):
        """
        Arguments:
            G: MaxCut graph, Networkx
            gamma, beta: list[float]

        Returns: MaxCut energy expectation
        """

        total_E = 0
        
        for node in G.nodes():
            if G.nodes[node]['weight'] is not None:
                total_E += self._get_node_energy(G, gamma, beta, node)
        
        for edge in G.edges():
            total_E += self._get_edge_energy(G, gamma, beta, edge)
        
        C = self._post_process_energy(G, total_E)
        return C
    
    def _post_process_energy(self, G, E):
        if np.any(np.abs(np.imag(E)) > 1e-6):
            print(f"Warning: Energy result imaginary part was: {np.imag(E)}")
        
        ans = np.real(E)
        if 'offset' in G.graph:
            ans += G.graph['offset']
        return ans

class QtreeIsingQAOASimulator(IsingQAOASimulator, QtreeSimulator):
    # Options:
    #   a) Use the hacky self._edge_energy_circuit(G, gamma, beta, (node, node)).
    #   b) Create node versions of _edge_energy_circuit, energy_expectation_lightcone, get_edge_subgraph, energy_expectation.
    #   c) Modify _edge_energy_circuit, energy_expectation_lightcone, get_edge_subgraph, energy_expectation to work with both nodes and edges ("terms").
        
    # (a) is the simplest, but could lead to bugs in the future.
    # (b) repeats code, but easy to incorporate into the existing codebase without modifying it.
    # (c) is the cleanest.
    
    def _get_node_energy(self, G, gamma, beta, node):
        # # Call stack:
        # #   self._edge_energy_circuit(G, gamma, beta, (node, node))
        # #       composer.energy_expectation_lightcone((node, node))
        # #           utils.get_edge_subgraph(G, (node, node)), len(gamma))
        # #               nodes_group_by_distance(G, (node, node), dist)
        # #                   G.subgraph(all_nodes).copy()
        # #                   # This gets called with an additional copy of node in all_nodes. Networkx handles this properly.
        # #           composer.energy_expectation(i,j)
        # #               self.energy_edge(i, j)
        # #               # This has been redefined to handle i=j properly
        
        # circuit = self._edge_energy_circuit(G, gamma, beta, (node, node)) # Approach (a) specific
        circuit = self._node_energy_circuit(G, gamma, beta, node) # Approach (b) specific
        weight = G.nodes[node]['weight']
        return weight*self.simulate(circuit)
    
    def _get_edge_energy(self, G, gamma, beta, edge):
        circuit = self._edge_energy_circuit(G, gamma, beta, edge)
        weight = G.get_edge_data(*edge)['weight']
        return weight*self.simulate(circuit)

class CirqIsingQAOASimulator(IsingQAOASimulator, CirqSimulator):
    def _get_node_energy(self, G, gamma, beta, node):
        self.max_tw = 25
        if not hasattr(self, '_warned'):
            print('Warning: the energy calculation is not yet implemented')
            self._warned = True
        # circuit = self._edge_energy_circuit(G, gamma, beta, (node, node)) # Approach (a) specific
        circuit = self._node_energy_circuit(G, gamma, beta, node) # Approach (b) specific
        trial_result = self.simulate(circuit)
        weight = G.nodes[node]['weight']
        
        return weight*np.sum(trial_result.state_vector())
    
    def _get_edge_energy(self, G, gamma, beta, edge):
        self.max_tw = 25
        if not hasattr(self, '_warned'):
            print('Warning: the energy calculation is not yet implemented')
            self._warned = True
        circuit = self._edge_energy_circuit(G, gamma, beta, edge)
        trial_result = self.simulate(circuit)
        weight = G.get_edge_data(*edge)['weight']
        
        return weight*np.sum(trial_result.state_vector())

class CirqIsingQAOAComposer(IsingQAOAComposer):
    def _get_builder_class(self):
        return CirqBuilder

class QiskitIsingQAOAComposer(IsingQAOAComposer):
    def _get_builder_class(self):
        return QiskitBuilder

class QtreeIsingQAOAComposer(IsingQAOAComposer):
    def _get_builder_class(self):
        return QtreeBuilder

class CirqIsingZZQAOAComposer(IsingZZQAOAComposer):
    def _get_builder_class(self):
        return CirqBuilder

class QiskitIsingZZQAOAComposer(IsingZZQAOAComposer):
    def _get_builder_class(self):
        return QiskitBuilder

class QtreeIsingZZQAOAComposer(IsingZZQAOAComposer):
    def _get_builder_class(self):
        return QtreeBuilder