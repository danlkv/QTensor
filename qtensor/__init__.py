# -- configure logging
import sys
from loguru import logger as log
log.remove()
log.add(sys.stderr, level='INFO')
# --
from qtensor import utils
from qtensor.utils import get_edge_subgraph
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from .CircuitComposer import QAOAComposer, OldQAOAComposer, ZZQAOAComposer, WeightedZZQAOAComposer, CircuitComposer
from .OpFactory import CirqBuilder, QtreeBuilder, QiskitBuilder, TorchBuilder
from .OpFactory import QtreeFullBuilder
from qtensor.Simulate import CirqSimulator, QtreeSimulator
from qtensor.QAOASimulator import QAOAQtreeSimulator
from qtensor.QAOASimulator import QAOACirqSimulator
from qtensor.QAOASimulator import QAOAQtreeSimulatorSymmetryAccelerated
from qtensor.FeynmanSimulator import FeynmanSimulator, FeynmanMergedSimulator
from qtensor import contraction_backends
from qtensor.contraction_backends import PerfNumpyBackend, NumpyBackend
from qtensor import simplify_circuit
from qtensor.simplify_circuit import simplify_qtree_circuit
from qtensor import optimisation
from qtensor import merged_indices
from qtensor import problems
from qtensor import MergedSimulator
from qtensor import tools

class CirqQAOAComposer(QAOAComposer):
    def _get_builder_class(self):
        return CirqBuilder

class QiskitQAOAComposer(QAOAComposer):
    def _get_builder_class(self):
        return QiskitBuilder

class QtreeQAOAComposer(QAOAComposer):
    def _get_builder_class(self):
        return QtreeBuilder

import random
import copy
    
class NoisyQAOAQtreeSimulator(QAOAQtreeSimulator):
    def __init__(self, composer, p_noise, *args, **kwargs):
        super().__init__(composer, *args, **kwargs)
        self.p_noise = p_noise
        self.noise_gates = None  # Will be generated when _edge_energy_circuit is called
            
    def assign_noise(self):
        r = np.random.rand()
        if r < 1 - self.p_noise:
            return 'I'  # Identity gate, no change needed
        elif r < 1 - (2/3)*self.p_noise:
            return 'X'
        elif r < 1 - (1/3)*self.p_noise:
            return 'Y'
        else:
            return 'Z'

    def generate_noisy_circuit(self, original_circuit):
        noisy_circuit = []
        for gate in original_circuit:
            noise_gates_for_qubits = [self.assign_noise() for _ in gate.qubits]
            noisy_circuit.append((gate, noise_gates_for_qubits))
        return noisy_circuit
            
    def energy_expectation(self, G, gamma, beta):
        """
        Arguments:
            G: MaxCut graph, Networkx
            gamma, beta: list[float]

        Returns: MaxCut energy expectation
        """
        composer = self.composer(G, gamma=gamma, beta=beta)
        composer.ansatz_state()
        qaoa_circuit = composer.circuit
        
        self.noisy_circuit = self.generate_noisy_circuit(qaoa_circuit)

        total_E = 0

        with tqdm(total=G.number_of_edges(), desc='Edge iteration', ) as pbar:
            for i, edge in enumerate(G.edges()):
                E = self._get_edge_energy(G, gamma, beta, edge)
                # debt
                pbar.set_postfix(Treewidth=self.optimizer.treewidth)
                pbar.update(1)
                total_E += E

            if self.profile:
                # debt
                print(self.backend.gen_report())
        C = self._post_process_energy(G, total_E)
        return C
    

    def construct_noisy_edge_circuit(self, edge_circuit, composer):
        new_circuit = []

        # Create a lookup dictionary for noise gates
        noise_lookup = {str(gate): noise for gate, noise in self.noisy_circuit}

        for gate in edge_circuit:
            new_circuit.append(gate)

            noise_gates_for_qubits = noise_lookup.get(str(gate), None)
            if noise_gates_for_qubits:
                for qubit, noise_gate_type in zip(gate.qubits, noise_gates_for_qubits):
                    if noise_gate_type == 'I':
                        continue  # Skip identity gates
                    # Apply the noise gate to the chosen qubit
                    if noise_gate_type == 'X':
                        new_circuit.append(composer.operators.X(qubit))
                    elif noise_gate_type == 'Y':
                        new_circuit.append(composer.operators.Y(qubit))
                    elif noise_gate_type == 'Z':
                        new_circuit.append(composer.operators.Z(qubit))
        return new_circuit


    def _get_edge_energy(self, G, gamma, beta, edge):
        composer = self.composer(G, gamma=gamma, beta=beta)
        composer.cone_ansatz(edge)

        edge_circuit = copy.deepcopy(composer.circuit)
        noisy_edge_circuit = self.construct_noisy_edge_circuit(edge_circuit, composer)
#         print('noisy edge circuit', noisy_edge_circuit)

        composer.circuit = noisy_edge_circuit
        composer.builder.inverse()
        inverse_circuit = composer.builder.circuit

        u, v = composer.qubit_map[edge[0]], composer.qubit_map[edge[1]]
        noisy_edge_circuit.extend([composer.operators.Z(u), composer.operators.Z(v)])
        noisy_edge_circuit.extend(inverse_circuit)
#         print('\n')
#         print('final noisy edge circuit', noisy_edge_circuit)

        cdata = self._edge_cache.get(edge)
        if cdata is not None:
            peo, width = cdata
            return self.simulate_batch(noisy_edge_circuit, batch_vars=0, peo=peo)
        else:
            return self.simulate(noisy_edge_circuit)


class NoisyQtreeQAOAComposer(QtreeQAOAComposer):
    def __init__(self, graph, noise_prob=0.05, *args, **kwargs):
        super().__init__(graph, *args, **kwargs)
        self.noise_prob = noise_prob
    
    
    def apply_depolarizing_noise(self, qubit):
        """ Apply depolarizing noise with probability noise_prob. """
        if np.random.rand() < self.noise_prob:
            print(f'applying random gate (noise prob = {self.noise_prob})')
            # Choose a random gate from a predefined list (e.g., X, Y, Z)
            random_gate = np.random.choice([self.operators.X, self.operators.Y, self.operators.Z])
            self.apply_gate(random_gate, qubit)
            
            
    def x_term(self, u, beta):
        #self.circuit.append(self.operators.H(u))
        self.apply_gate(self.operators.XPhase, u, alpha=2*beta)
        #self.circuit.append(self.operators.H(u))
        self.apply_depolarizing_noise(u)
        
            
    def append_zz_term(self, q1, q2, gamma):
        self.apply_gate(self.operators.cX, q1, q2)
        self.apply_depolarizing_noise(q1)
        self.apply_depolarizing_noise(q2)
        
        self.apply_gate(self.operators.ZPhase, q2, alpha=2*gamma)
        self.apply_depolarizing_noise(q2)
        
        self.apply_gate(self.operators.cX, q1, q2)
        self.apply_depolarizing_noise(q1)
        self.apply_depolarizing_noise(q2)
        
        
    def layer_of_Hadamards(self):
        for q in self.qubits:
            self.apply_gate(self.operators.H, q)
            self.apply_depolarizing_noise(q)
        self.apply_gate(self.operators.Y, self.qubits[0])
        self.apply_depolarizing_noise(self.qubits[0])
        
    def energy_expectation(self, i, j):
        # Will need to deprecate stateful API and return the circuit
        self.cone_ansatz(edge=(i, j))
        composer_copy = copy.deepcopy(self)
        
        self.energy_edge(i, j)
        first_part = self.builder.circuit
        # self.builder.reset()

        # self.cone_ansatz(edge=(i, j))
        composer_copy.builder.inverse()
        second_part = composer_copy.builder.circuit

        self.circuit = first_part + second_part
        
#         print('first part:', first_part)
#         print('second part:', second_part)
#         print('\n')
        # print(self.circuit)
        

class QtreeFullQAOAComposer(QAOAComposer):
    def _get_builder_class(self):
        return QtreeFullBuilder

class OldQtreeQAOAComposer(OldQAOAComposer):
    def _get_builder_class(self):
        return QtreeBuilder

class ZZQtreeQAOAComposer(ZZQAOAComposer):
    def _get_builder_class(self):
        return QtreeBuilder

class ZZQtreeFullQAOAComposer(ZZQAOAComposer):
    def _get_builder_class(self):
        return QtreeFullBuilder

class WeightedZZQtreeQAOAComposer(WeightedZZQAOAComposer):
    def _get_builder_class(self):
        return QtreeBuilder

class SimpZZQtreeComposer(ZZQtreeQAOAComposer):
    @property
    def circuit(self):
        return simplify_qtree_circuit(self.builder.circuit)
    @circuit.setter
    def circuit(self, circuit):
        self.builder.circuit = circuit

class TorchQAOAComposer(ZZQtreeQAOAComposer):
    def _get_builder_class(self):
        return TorchBuilder

#DefaultQAOAComposer = SimpZZQtreeComposer
DefaultQAOAComposer = ZZQtreeQAOAComposer
WeightedQAOAComposer = WeightedZZQtreeQAOAComposer


# deprecated
CCQtreeQAOAComposer = ZZQtreeQAOAComposer

def QAOA_energy(G, gamma, beta, n_processes=0):
    sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    if n_processes:
        res = sim.energy_expectation_parallel(G, gamma=gamma, beta=beta
            ,n_processes=n_processes
        )
    else:
        res = sim.energy_expectation(G, gamma=gamma, beta=beta)
    return res


from . import toolbox
