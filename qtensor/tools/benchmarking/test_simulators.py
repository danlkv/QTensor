import qtensor
import numpy as np
import qtensor.tests
import pytest
from qtensor.tools.benchmarking import simulators
from qtensor.tools.benchmarking.simulators import SIMULATORS, BenchSimulator
from functools import lru_cache

TEST_SIMS = ['qtensor', 'qtensor_merged', 'acqdp', 'quimb']

try:
    import acqdp
except:
    TEST_SIMS.remove('acqdp')

try:
    import quimb, cotengra
except:
    TEST_SIMS.remove('quimb')

@pytest.fixture(params=TEST_SIMS)
def simulator(request):
    sim = simulators.SIMULATORS[request.param]
    return sim()

P = 2

@pytest.fixture(params=[4])
def problem(request):
    return qtensor.tests.get_test_problem(n=request.param, p=P)


def reference_value(G, p):
    gamma, beta = simulators.get_test_gamma_beta(p=p)
    gamma = np.array(gamma)/np.pi
    beta = np.array(beta)/np.pi
    sim = qtensor.QAOAQtreeSimulator(qtensor.QtreeQAOAComposer)
    E = sim.energy_expectation(G, gamma=gamma, beta=beta)
    E = -2*E + G.number_of_edges()
    return E


def test_sim(simulator, problem):
    G, gamma, beta = problem
    p = P
    opts, ests, opt_time = simulator.optimize_qaoa_energy(G, p=p)
    assert all([type(x.width) in [float, int] for x in ests])
    assert all([type(x.mems) in [float, int] for x in ests])
    assert all([type(x.flops) in [float, int] for x in ests])
    max_w = max([x.width for x in ests])
    print('Max width', max_w)
    print('Optimization time', opt_time)

    result, time, mem = simulator.simulate_qaoa_energy(G, p, opts)
    print('result value', result)

    ref = reference_value(G, p)


    assert np.isclose(ref, result)

def test_time_budget(simulator: BenchSimulator):
    p = 2
    G, gamma, beta = qtensor.tests.get_test_problem(n=46, p=p)
    # quimb optimization is slow
    if isinstance(simulator, simulators.QuimbSimulator):
        G, gamma, beta = qtensor.tests.get_test_problem(n=6, p=p)



    print('Test time budget exceeded')
    simulator.set_max_time(.1)
    print('Optimize')
    with pytest.raises(simulators.TimeExceeded):
        opts, ests, opt_time = simulator.optimize_qaoa_energy(G, p=p)

    budget = 5
    simulator.set_max_time(budget)
    opts, ests, opt_time = simulator.optimize_qaoa_energy(G, p=p)
    assert opt_time < budget

    simulator.set_max_time(.05)
    # quimb simulation is fast
    if isinstance(simulator, simulators.QuimbSimulator):
        simulator.set_max_time(.005)

    print('Simulate')
    with pytest.raises(simulators.TimeExceeded):
        result, time, mem = simulator.simulate_qaoa_energy(G, p, opts)

    simulator.set_max_time(budget)
    result, time, mem = simulator.simulate_qaoa_energy(G, p, opts)
    assert time < budget
