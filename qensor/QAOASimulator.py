from qensor.Simulate import Simulator, QtreeSimulator
from qensor.utils import get_edge_subgraph
import numpy as np
import networkx as nx
from tqdm import tqdm

class QAOASimulator(Simulator):
    def __init__(self, composer, profile=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.composer = composer
        self.profile = profile

    def energy_expectation(self, G, gamma, beta):
        """
        Arguments:
            G: MaxCut graph, Networkx
            gamma, beta: list[float]

        Returns: MaxCut energy expectation
        """

        total_E = 0

        for edge in tqdm(G.edges(), 'Edge iteration'):
            i,j = edge
            # TODO: take only a neighbourhood part of the graph
            graph = get_edge_subgraph(G, edge, len(gamma))
            mapping = {v:i for i, v in enumerate(graph.nodes())}
            graph = nx.relabel_nodes(graph, mapping, copy=True)

            composer = self.composer(
                graph=graph, gamma=gamma, beta=beta)

            i,j = mapping[i], mapping[j]
            composer.energy_expectation(i,j)

            result = self.simulate(composer.circuit)
            E = result
            if self.profile:
                print(self.backend.gen_report())
            total_E += E

        E = total_E
        #print(composer.circuit)
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
        C = (Ed - E)/2
        return C

class QAOAQtreeSimulator(QAOASimulator, QtreeSimulator):
    pass
