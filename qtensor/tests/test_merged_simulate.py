import numpy as np
import networkx as nx
import pytest
from functools import lru_cache
import pyrofiler as prof

import qtensor
from qtensor import QtreeQAOAComposer
from qtensor.contraction_backends import PerfNumpyBackend

from qtensor.Simulate import CirqSimulator, QtreeSimulator

@lru_cache
def get_test_problem(n=10, p=2, d=3, type='random'):
    print('Test problem: n, p, d', n, p, d)
    if type == 'random':
        G = nx.random_regular_graph(d, n)
    elif type == 'grid2d':
        G = nx.grid_2d_graph(n,n)
    elif type == 'line':
        G = nx.Graph()
        G.add_edges_from(zip(range(n-1), range(1, n)))
    G = nx.convert_node_labels_to_integers(G)
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p
    return G, gamma, beta

@pytest.fixture
def test_problem(request):
    n, p, d, type = request.param
    return get_test_problem(n, p, d, type)

paramtest = [
    # n, p, degree, type
     [4, 4, 3, 'random']
    ,[10, 5, 2, 'random']
    ,[20, 4, 3, 'random']
    ,[3, 3, 0, 'grid2d']
    ,[8, 4, 0, 'line']
]

@pytest.mark.parametrize('test_problem', paramtest ,indirect=True)
def test_merged_ix(test_problem):
    G, gamma, beta = get_test_problem()
    G, gamma, beta = test_problem
    print('default', G.number_of_nodes(), G.number_of_edges(), len(gamma))
    comp = qtensor.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
    comp.ansatz_state()
    opt  = qtensor.optimisation.RGreedyOptimizer(temp=0.01, repeats=5)
    #opt  = qtensor.optimisation.GreedyOptimizer()

    backend = qtensor.contraction_backends.NumpyBackend()
    m_sim = qtensor.MergedSimulator.MergedSimulator(backend=backend, optimizer=opt)
    sim = qtensor.QtreeSimulator(backend=backend, optimizer=opt)

    with prof.timing('Default simulator time:'):
        amp = sim.simulate(comp.circuit)
    print('tw2', opt.treewidth)

    with prof.timing('Merged simulator time:'):
        m_amp = m_sim.simulate(comp.circuit)
    print('tw1', opt.treewidth)
    assert np.allclose(amp, m_amp)
