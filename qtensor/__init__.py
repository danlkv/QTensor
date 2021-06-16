# -- configure logging
import sys
from functools import lru_cache
from loguru import logger as log
log.remove()
log.add(sys.stderr, level='INFO')
# --
from qtensor import utils
from qtensor.utils import get_edge_subgraph
import networkx as nx

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
