import numpy as np

from .CircuitComposer import QAOAComposer
from .OpFactory import CirqCreator, QtreeCreator
from qensor.Simulate import CirqSimulator, QtreeSimulator

class CirqQAOAComposer(QAOAComposer, CirqCreator):
    pass

class QtreeQAOAComposer(QAOAComposer, QtreeCreator):
    def energy_expectation(self):
        G = self.graph
        self.ansatz_state()
        for i,j in G.edges():
            u, v = self.qubits[i], self.qubits[j]
            self.energy_edge(u, v)

        beta, gamma = self.params['beta'], self.params['gamma']
        conjugate = QtreeQAOAComposer(G, beta=beta, gamma=gamma)
        conjugate.ansatz_state()
        conjugate = [g.dagger_me() for g in conjugate.circuit]

        self.circuit = self.circuit + list(reversed(conjugate))
        return self.circuit

def QAOA_energy(G, gamma, beta):
    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.energy_expectation()

    print(composer.circuit)
    sim = QtreeSimulator()
    result = sim.simulate(composer.circuit)
    print(result.data)
    E = result.data
    if np.imag(E)>1e-6:
        print(f"Warning: Energy result imaginary part was: {np.imag(E)}")

    return np.real(E)

