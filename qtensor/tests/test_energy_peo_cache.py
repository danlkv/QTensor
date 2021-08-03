import numpy as np
from qtensor.tests import get_test_problem
import pyrofiler as prof
from qtensor import QAOAQtreeSimulator
from qtensor import DefaultQAOAComposer

def test_peo_cache():
    G, gamma, beta = get_test_problem(10, 2, 4, 'random')
    p = len(gamma)
    print('default', G.number_of_nodes(), G.number_of_edges(), len(gamma))
    sim = QAOAQtreeSimulator(DefaultQAOAComposer)
    with prof.timing('QTensor energy time') as t1:
        E1 = sim.energy_expectation(G, gamma=gamma, beta=beta)

    with prof.timing() as topt:
        sim.optimize_lightcones(G, p)
    with prof.timing('Qiskit energy time') as t2:
        E2 = sim.energy_expectation(G, gamma=gamma, beta=beta)
    assert np.isclose(E1, E2)
    assert t2.result < t1.result - topt.result*.5