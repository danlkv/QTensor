import numpy as np
from functools import lru_cache
import networkx as nx
import pytest
import qtensor

@pytest.fixture
def test_problem():
    n, p, d = 10, 4, 1
    G1 = nx.random_regular_graph(d, n)
    G2 = nx.random_regular_graph(d, n)
    G = nx.union(G1, G2, rename=('a', 'b'))
    print('Gnodes', G.nodes(), G.edges())
    gamma = tuple([np.pi/5]*p)
    beta = tuple([np.pi/2]*p)
    return G, gamma, beta


@pytest.fixture
def ordering_algo(request):
    alg = request.param
    return qtensor.toolbox.get_ordering_algo(alg)

ALGOS = [
    #'greedy',
    'naive',
    'rgreedy_0.02_15',
    #'kahypar',
]

@lru_cache()
def reference(test_problem):
    G, gamma, beta = test_problem
    comp = qtensor.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
    comp.ansatz_state()
    sim = qtensor.QtreeSimulator()
    return sim.simulate_batch(comp.circuit, batch_vars=6)

@pytest.mark.parametrize('ordering_algo', ALGOS, indirect=True)
def test_simulate_batch(test_problem, ordering_algo):
    G, gamma, beta = test_problem
    comp = qtensor.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
    comp.ansatz_state()

    sim = qtensor.QtreeSimulator(optimizer=ordering_algo)
    res = sim.simulate_batch(comp.circuit, batch_vars=6)
    refr = reference(test_problem)
    assert np.allclose(res, refr)