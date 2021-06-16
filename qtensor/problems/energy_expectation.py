from qtensor.Simulate import Simulator
import qtensor
import numpy as np

class EnergyExpectation():
    def __init__(self, operators, qubits, coefficients, simulator):
        """
        Args:
            operators: a list of elements of qtree.OpFactory
            qubits: list of tuples of ints
            coefficients: list of floats
            simulator: instance of qtensor.Simulator
        """

        self.operators = operators
        self.qubits = qubits
        self.coefficients = coefficients
        self._simulator = simulator

    def set_simulator(self, simulator):
        self._simulator = simulator

    def preprocess(self, composer, ordering_algo='default'):
        peos = []
        widths = []
        for op, q in zip(self.operators, self.qubits):
            circ = composer.expectation(op, *q)
            tn = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
            opt = qtensor.toolbox.get_ordering_algo(ordering_algo)
            peo, _ = opt.optimize(tn)
            peos.append(peo)
            widths.append(opt.treewidth)

        return peos, widths

    def get_complexity(self, composer, **kwargs):
        peos, widths = self.preprocess(composer, **kwargs)
        return max(widths)

    def simulate(self, composer):
        contribs = []
        for op, q in zip(self.operators, self.qubits):
            circ = composer.expectation(op, *q)
            contrib = self._simulator.simulate(circ)
            assert len(contrib) == 1
            contribs.append(contrib[0])

        return sum(x*c for x, c in zip(contribs, self.coefficients))


