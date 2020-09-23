# -- configure logging
import sys
from loguru import logger as log
log.remove()
log.add(sys.stderr, level='INFO')
# --
from qtensor.utils import get_edge_subgraph
import networkx as nx

from .CircuitComposer import QAOAComposer
from .OpFactory import CirqBuilder, QtreeBuilder, QiskitBuilder
from qtensor.Simulate import CirqSimulator, QtreeSimulator
from qtensor.QAOASimulator import QAOAQtreeSimulator
from qtensor.QAOASimulator import QAOACirqSimulator
from qtensor.FeynmanSimulator import FeynmanSimulator
from qtensor.ProcessingFrameworks import PerfNumpyBackend, NumpyBackend

class CirqQAOAComposer(QAOAComposer):
    def _get_builder_class(self):
        return CirqBuilder

class QiskitQAOAComposer(QAOAComposer):
    def _get_builder_class(self):
        return QiskitBuilder

class QtreeQAOAComposer(QAOAComposer):
    def _get_builder_class(self):
        return QtreeBuilder

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
