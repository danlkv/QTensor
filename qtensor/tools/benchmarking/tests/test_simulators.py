import qtensor
from qtensor.tools.benchmarking import simulators

def  test_qtensor_energy():
    sim = simulators.QtensorSimulator()
    graph = qtensor.toolbox.random_graph(nodes=18, degree=3, seed=11)
    opts, ests, opt_time = sim.optimize_qaoa_energy(graph, p=3, ordering_algo='rgreedy_0.02_10')
    res, time, mem = sim.simulate_qaoa_energy(graph, p=3, opt = opts)

    assert time>0
    assert mem<1e7
