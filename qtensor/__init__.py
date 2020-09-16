# -- configure logging
import sys
from loguru import logger as log
log.remove()
log.add(sys.stderr, level='INFO')
# --
from qtensor.utils import get_edge_subgraph
import networkx as nx

from .CircuitComposer import QAOAComposer
from .OpFactory import CirqCreator, QtreeCreator
from .OpFactory import CirqCreator, QtreeCreator, QiskitCreator
from qtensor.Simulate import CirqSimulator, QtreeSimulator
from qtensor.QAOASimulator import QAOAQtreeSimulator
from qtensor.FeynmanSimulator import FeynmanSimulator
from qtensor.ProcessingFrameworks import PerfNumpyBackend, NumpyBackend

class CirqQAOAComposer(QAOAComposer, CirqCreator):
    pass

class QiskitQAOAComposer(QAOAComposer, QiskitCreator):
    pass

class QtreeQAOAComposer(QAOAComposer, QtreeCreator):
    def energy_expectation(self, i, j):
        G = self.graph
        self.ansatz_state()
        self.energy_edge(i, j)

        beta, gamma = self.params['beta'], self.params['gamma']
        conjugate = QtreeQAOAComposer(G, beta=beta, gamma=gamma)
        conjugate.ansatz_state()
        conjugate = [g.dagger_me() for g in conjugate.circuit]

        self.circuit = self.circuit + list(reversed(conjugate))
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
        composer = QtreeQAOAComposer(graph, beta=beta, gamma=gamma)
        composer.energy_expectation(i,j)
        self.circuit = composer.circuit
        # return composer


def QAOA_energy(G, gamma, beta, n_processes=0):
    sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    if n_processes:
        res = sim.energy_expectation_parallel(G, gamma=gamma, beta=beta
            ,n_processes=n_processes
        )
    else:
        res = sim.energy_expectation(G, gamma=gamma, beta=beta)
    return res
