from qensor.Simulate import Simulator, QtreeSimulator
from qensor.utils import get_edge_subgraph
import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
from loguru import logger as log

class QAOASimulator(Simulator):
    def __init__(self, composer, profile=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.composer = composer
        self.profile = profile

    def _get_edge_energy(self, G, gamma, beta, edge):
        i,j = edge
        # TODO: take only a neighbourhood part of the graph
        graph = get_edge_subgraph(G, edge, len(gamma))
        log.debug('Subgraph nodes: {}, edges: {}', graph.number_of_nodes(), graph.number_of_edges())
        graph = get_edge_subgraph(G, edge, len(gamma))
        mapping = {v:i for i, v in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping, copy=True)

        composer = self.composer(
            graph=graph, gamma=gamma, beta=beta)

        i,j = mapping[i], mapping[j]
        composer.energy_expectation(i,j)

        return self.simulate(composer.circuit)

    def _post_process_energy(self, G, E):
        if np.imag(E)>1e-6:
            print(f"Warning: Energy result imaginary part was: {np.imag(E)}")

        """
        C = sum(CC)
        2*CC = 1 - ZZ
        2*C = sum(1-CC)
        2*C = Ed - sum(CC)
        C = (Ed - E)/2
        """
        E = np.real(E)

        Ed = G.number_of_edges()
        return (Ed - E)/2


    def energy_expectation(self, G, gamma, beta):
        """
        Arguments:
            G: MaxCut graph, Networkx
            gamma, beta: list[float]

        Returns: MaxCut energy expectation
        """

        total_E = 0

        for edge in tqdm(G.edges(), 'Edge iteration'):
            E = self._get_edge_energy(G, gamma, beta, edge)
            if self.profile:
                print(self.backend.gen_report())
            total_E += E

        C = self._post_process_energy(G, total_E)
        return C


    def _parallel_unit_edge(self, args):
        return self._get_edge_energy(*args)

    def energy_expectation_parallel(self, G, gamma, beta, n_processes=4):
        """
        Arguments:
            G: MaxCut graph, Networkx
            gamma, beta: list[float]

        Returns: MaxCut energy expectation
        """
        args = [(G, gamma, beta, edge) for edge in G.edges()]

        with Pool(n_processes) as p:

           r = list(tqdm(p.imap(self._parallel_unit_edge, args), total=G.number_of_edges()))
           total_E = sum(r)
        C = self._post_process_energy(G, total_E)

        return C


class QAOAQtreeSimulator(QAOASimulator, QtreeSimulator):
    pass
