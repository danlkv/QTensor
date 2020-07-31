from .CircuitComposer import QAOAComposer
from .OpFactory import CirqCreator, QtreeCreator
from qensor.Simulate import CirqSimulator, QtreeSimulator
import qtree
from qensor.QAOASimulator import QAOAQtreeSimulator
from qensor.FeynmanSimulator import FeynmanSimulator
from qensor.ProcessingFrameworks import PerfNumpyBackend, NumpyBackend

class CirqQAOAComposer(QAOAComposer, CirqCreator):
    pass


class QtreeQAOAComposer(QAOAComposer, QtreeCreator):

    def set_operators(self, operators):
        if operators == "diagonal":
            self.operators = qtree.operators
        elif operators == "full_matrix":
            self.operators = qtree.operators_full_matrix

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

def QAOA_energy(G, gamma, beta, n_processes=0):
    sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    if n_processes:
        res = sim.energy_expectation_parallel(G, gamma=gamma, beta=beta
            ,n_processes=n_processes
        )
    else:
        res = sim.energy_expectation(G, gamma=gamma, beta=beta)
    return res
