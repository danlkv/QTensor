import numpy as _np
import networkx as _nx

from qtensor.CircuitComposer import QAOAComposer as _QAOAComposer
from qtensor.Simulate import QtreeSimulator as _QtreeSimulator
from qtensor.Simulate import CirqSimulator as _CirqSimulator
from qtensor.QAOASimulator import QAOASimulator as _QAOASimulator
from qtensor.OpFactory import CirqBuilder as _CirqBuilder
from qtensor.OpFactory import QiskitBuilder as _QiskitBuilder
from qtensor.OpFactory import QtreeBuilder as _QtreeBuilder
# from qtensor.optimisation.Optimizer import DefaultOptimizer

def qubo_to_graph(qubo):
    n = qubo.get_num_binary_vars()
    qubo_objective = qubo.objective
    
    qubo_constant = qubo_objective.constant
    qubo_linear = qubo_objective.linear.to_dict()
    qubo_quadratic = qubo_objective.quadratic.to_dict()
    
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
    
    G = _nx.Graph(offset=graph_constant)
    G.add_nodes_from(range(n))
    for i in graph_linear:
        G.nodes[i]['weight'] = graph_linear[i]
        # G.add_edge(i, i, weight=graph_linear[i])
    
    for i, j in graph_quadratic:
        G.add_edge(i, j, weight=graph_quadratic[(i,j)])
    
    return G

class _IsingQAOAComposer(_QAOAComposer):
    def energy_edge(self, i, j):
        u, v = self.qubits[i], self.qubits[j]
        self.apply_gate(self.operators.Z, u)
        if i != j:
            self.apply_gate(self.operators.Z, v)
    
    def cost_operator_circuit(self, gamma, edges=None):
        for i, w in self.graph.nodes.data('weight'):
            if w is None:
                continue
            u = self.qubits[i]
            self.apply_gate(self.operators.ZPhase, u, alpha=2*gamma*w)
        
        for i, j, w in self.graph.edges.data('weight'):
            u, v = self.qubits[i], self.qubits[j]
            self.append_zz_term(u, v, gamma*w)
        
        # for i, j, w in self.graph.edges.data('weight', default=1):
        #     u, v = self.qubits[i], self.qubits[j]
        #     if i != j:
        #         self.append_zz_term(u, v, gamma*w)
        #     else:
        #         self.apply_gate(self.operators.ZPhase, u, alpha=2*gamma*w)

class _IsingZZQAOAComposer(_IsingQAOAComposer):
    def append_zz_term(self, q1, q2, gamma):
        self.apply_gate(self.operators.ZZ, q1, q2, alpha=2*gamma)

class CirqIsingQAOAComposer(_IsingQAOAComposer):
    def _get_builder_class(self):
        return _CirqBuilder

class QiskitIsingQAOAComposer(_IsingQAOAComposer):
    def _get_builder_class(self):
        return _QiskitBuilder

class QtreeIsingQAOAComposer(_IsingQAOAComposer):
    def _get_builder_class(self):
        return _QtreeBuilder

class CirqIsingZZQAOAComposer(_IsingZZQAOAComposer):
    def _get_builder_class(self):
        return _CirqBuilder

class QiskitIsingZZQAOAComposer(_IsingZZQAOAComposer):
    def _get_builder_class(self):
        return _QiskitBuilder

class QtreeIsingZZQAOAComposer(_IsingZZQAOAComposer):
    def _get_builder_class(self):
        return _QtreeBuilder

class _IsingQAOASimulator(_QAOASimulator):
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
        if _np.imag(E)>1e-6:
            print(f"Warning: Energy result imaginary part was: {_np.imag(E)}")
        
        ans = _np.real(E)
        if 'offset' in G.graph:
            ans += G.graph['offset']
        return ans

class QtreeIsingQAOASimulator(_IsingQAOASimulator, _QtreeSimulator):
    def _get_node_energy(self, G, gamma, beta, node):
        circuit = self._edge_energy_circuit(G, gamma, beta, (node, node))
        weight = G.nodes[node]['weight']
        return weight*self.simulate(circuit)
    
    def _get_edge_energy(self, G, gamma, beta, edge):
        circuit = self._edge_energy_circuit(G, gamma, beta, edge)
        weight = G.get_edge_data(*edge)['weight']
        return weight*self.simulate(circuit)

class CirqIsingQAOASimulator(_IsingQAOASimulator, _CirqSimulator):
    def _get_edge_energy(self, G, gamma, beta, edge):
        self.max_tw = 25
        if not hasattr(self, '_warned'):
            print('Warning: the energy calculation is not yet implemented')
            self._warned = True
        circuit = self._edge_energy_circuit(G, gamma, beta, edge)
        trial_result = self.simulate(circuit)
        weight = G.get_edge_data(*edge)['weight']
        
        return weight*_np.sum(trial_result.state_vector())