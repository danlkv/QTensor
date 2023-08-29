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
    
class NoisyQtreeQAOAComposer(QtreeQAOAComposer):
    def __init__(self, graph, noise_prob=0.5, *args, **kwargs):
        super().__init__(graph, *args, **kwargs)
        self.noise_prob = noise_prob
    
    def apply_depolarizing_noise(self, qubit, noise_prob=0.5):
        """ Apply depolarizing noise with probability noise_prob. """
        if np.random.rand() < self.noise_prob:
            print('applying random gate')
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
