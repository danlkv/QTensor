import numpy as np
import networkx as nx
import pytest
from functools import lru_cache
import pyrofiler as prof

import qtensor
from qtensor import QtreeQAOAComposer
from qtensor.contraction_backends import PerfNumpyBackend

from qtensor.Simulate import CirqSimulator, QtreeSimulator
from qtensor.tests import get_test_problem

paramtest = [
    # n, p, degree, type
     [4, 4, 3, 'random']
   , [10, 5, 2, 'random']
   , [20, 4, 3, 'random']
   , [3, 3, 0, 'grid2d']
   , [8, 4, 0, 'line']
]

@pytest.fixture(params=paramtest)
def test_problem(request):
    n, p, d, type = request.param
    return get_test_problem(n, p, d, type)

@pytest.fixture(params=['einsum'])
def backend(request):
    backend = qtensor.contraction_backends.get_backend(request.param)
    return backend

def test_merged_ix(test_problem, backend):
    G, gamma, beta = get_test_problem()
    G, gamma, beta = test_problem
    print('default', G.number_of_nodes(), G.number_of_edges(), len(gamma))
    comp = qtensor.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
    comp.ansatz_state()
    opt  = qtensor.optimisation.RGreedyOptimizer(temp=0.01, repeats=5)
    #opt  = qtensor.optimisation.GreedyOptimizer()

    m_sim = qtensor.MergedSimulator.MergedSimulator(backend=backend, optimizer=opt)
    sim = qtensor.QtreeSimulator(backend=backend, optimizer=opt)

    with prof.timing('Default simulator time:'):
        amp = sim.simulate(comp.circuit)
    print('tw2', opt.treewidth)

    with prof.timing('Merged simulator time:'):
        m_amp = m_sim.simulate(comp.circuit)

    assert np.allclose(amp, m_amp)


def test_merged_ix_sliced(test_problem, backend):
    G, gamma, beta = get_test_problem()
    G, gamma, beta = test_problem
    print('default', G.number_of_nodes(), G.number_of_edges(), len(gamma))
    comp = qtensor.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
    comp.ansatz_state()
    opt_par  = qtensor.optimisation.SlicesOptimizer(max_tw=5, max_slice=4)
    opt  = qtensor.optimisation.GreedyOptimizer()

    #opt  = qtensor.optimisation.GreedyOptimizer()

    m_sim = qtensor.FeynmanMergedSimulator(backend=backend, optimizer=opt_par)
    sim = qtensor.QtreeSimulator(backend=backend, optimizer=opt)

    with prof.timing('Default simulator time:'):
        amp = sim.simulate(comp.circuit)
    print('tw2', opt.treewidth)

    with prof.timing('Merged simulator time:'):
        m_amp = m_sim.simulate(comp.circuit)
    
    assert np.allclose(amp, m_amp)
