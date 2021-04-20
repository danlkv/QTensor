import qtensor
import qtree
import numpy as np
import networkx as nx
from qtensor import CirqQAOAComposer, QtreeQAOAComposer, DefaultQAOAComposer
from qtensor import QtreeSimulator
from functools import lru_cache
from qtensor.tests import get_test_problem

def get_simulator():
    return QtreeSimulator()

class ZxZ(qtree.operators.Gate):
    name = 'ZxZ'
    _changes_qubits=tuple( )
    def gen_tensor(self):
        return np.array([ [1, -1], [-1, 1] ])

def test_expectation():
    G, gamma, beta = get_test_problem()
    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)

    qubits = [(x, ) for x in [1, 2, 3]]
    ops = composer.builder.operators
    operators = [ops.H, ops.Z, ops.Y]
    coefficients = [1, .5, .5]
    simulator = get_simulator()
    problem = qtensor.problems.EnergyExpectation(operators, qubits, coefficients, simulator)

    composer.ansatz_state()
    res = problem.simulate(composer)
    print('result', res)
    assert res


    G, gamma, beta = get_test_problem(n=6, p=2)
    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    qubits = G.edges()
    print('qubits', qubits)

    operators = [ZxZ for _ in qubits]
    coefficients = [1 for _ in qubits]
    simulator = get_simulator()
    problem = qtensor.problems.EnergyExpectation(operators, qubits, coefficients, simulator)

    sim = qtensor.QAOAQtreeSimulator(QtreeQAOAComposer)
    reference = sim.energy_expectation(G, gamma, beta)

    E = problem.simulate(composer)
    E = np.real(E)

    Ed = G.number_of_edges()
    ee = (Ed - E)/2
    assert np.allclose(ee, reference)

def test_expectation_torch():
    pass
