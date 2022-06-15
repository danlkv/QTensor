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
    'greedy',
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
def test_simulate_batch_qaoa(test_problem, ordering_algo):
    G, gamma, beta = test_problem
    comp = qtensor.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
    comp.ansatz_state()

    sim = qtensor.QtreeSimulator(optimizer=ordering_algo)
    res = sim.simulate_batch(comp.circuit, batch_vars=6)
    refr = reference(test_problem)
    assert np.allclose(res, refr)

@pytest.mark.parametrize('ordering_algo', ALGOS, indirect=True)
def test_simulate_batch_1d(ordering_algo):
    from qtensor.OpFactory import CirqBuilder
    N = 10
    d = 5
    test_problem_qtensor = qtensor.tests.get_test_1d_problem(N, d=5)
    test_problem_circ = qtensor.tests.get_test_1d_problem(N, d=5, Builder=CirqBuilder)

    sim_qtr = qtensor.QtreeSimulator(optimizer=ordering_algo)
    sim_crq = qtensor.CirqSimulator()
    res_qtr = sim_qtr.simulate_batch(test_problem_qtensor, batch_vars=N)
    res_crq = sim_crq.simulate(test_problem_circ).final_state_vector
    print('Res', np.sum(np.square(np.abs(res_qtr))), res_qtr[0])
    print('Res_crq', np.sum(np.square(np.abs(res_qtr))), res_crq[0])

    assert np.allclose(res_qtr, res_crq, rtol=1e-5, atol=1e-7)
