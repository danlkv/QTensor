from qtensor.Simulate import Simulator
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

    def simulate(self, composer):
        contribs = []
        for op, q in zip(self.operators, self.qubits):
            circ = composer.expectation(op, *q)
            contrib = self._simulator.simulate(circ)
            assert contrib.size == 1
            contribs.append(contrib[0])

        return np.dot(contribs, self.coefficients)


