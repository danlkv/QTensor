import numpy as np

from .CircuitComposer import QAOAComposer
from .OpFactory import CirqCreator, QtreeCreator
from qensor.Simulate import CirqSimulator, QtreeSimulator
from qensor.ProcessingFrameworks import PerfNumpyBackend, NumpyBackend

class CirqQAOAComposer(QAOAComposer, CirqCreator):
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

def QAOA_energy(G, gamma, beta, profile=False):
    total_E = 0
    if profile:
        backend = PerfNumpyBackend()
    else:
        backend = NumpyBackend()
    sim = QtreeSimulator(bucket_backend=backend)

    for edge in G.edges():
        i,j = edge
        # TODO: take only a neighbourhood part of the graph
        composer = QtreeQAOAComposer(
            graph=G, gamma=gamma, beta=beta)
        composer.energy_expectation(i,j)
        result = sim.simulate(composer.circuit)
        E = result.data
        if profile:
            print(backend.gen_report())
        print(E)
        total_E += E

    E = total_E
    #print(result.data)
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

