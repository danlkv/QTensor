from .CircuitComposer import QAOAComposer
from .OpFactory import CirqCreator, QtreeCreator
from qensor.Simulate import CirqSimulator, QtreeSimulator
from qensor.QAOASimulator import QAOAQtreeSimulator
from qensor.FeynmanSimulator import FeynmanSimulator
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

