import qtensor
from qtensor import CirqQAOAComposer, QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.optimisation.Optimizer import OrderingOptimizer, TamakiTrimSlicing, TreeTrimSplitter
from qtensor.optimisation.Optimizer import SlicesOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor.FeynmanSimulator import FeynmanSimulator
import numpy as np
import networkx as nx
np.random.seed(42)
import pytest

def get_test_problem(n=14, p=2, d=3):

    G = nx.random_regular_graph(d, n)
    gamma, beta = [np.pi/3]*p, [np.pi/2]*p
    return G, gamma, beta

@pytest.mark.skip()
def test_tamaki_trimming_opt():
    G, gamma, beta = get_test_problem(34, p=3, d=3)

    composer = QtreeQAOAComposer(
            graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    sim = FeynmanSimulator()
    sim.optimizer = SlicesOptimizer
    result = sim.simulate(composer.circuit, batch_vars=3, tw_bias=7)
    print(result)


    sim.optimizer = TamakiTrimSlicing
    sim.opt_args = {'wait_time':5}
    result_tam = sim.simulate(composer.circuit, batch_vars=3, tw_bias=7)
    print(result_tam)
    assert np.allclose(result_tam , result)

def test_cost_estimation():
    def get_tw_costs(N):
        G, gamma, beta = get_test_problem(N, p=3, d=3)

        composer = QtreeQAOAComposer(
                graph=G, gamma=gamma, beta=beta)
        composer.energy_expectation_lightcone(list(G.edges())[0])
        tn = QtreeTensorNet.from_qtree_gates(composer.circuit)
        opt = OrderingOptimizer()
        peo, _= opt.optimize(tn)
        tw = opt.treewidth
        mems, flops = tn.simulation_cost(peo)
        print('Max memory=', max(mems), 'Total flops=', sum(flops), 'Treewidth=', tw)
        return tw, max(mems), sum(flops)

    for n in [ 10, 16, 26]:
        st1 = get_tw_costs(n+10)
        st2 = get_tw_costs(n)
        tw_diff = st1[0] - st2[0]
        log_mem_diff = np.log2(st1[1]/st2[1])
        log_flop_diff = np.log2(st1[1]/st2[1])
        print(f'{log_mem_diff=}, {log_flop_diff=}, {tw_diff=}')

        assert np.isclose(log_mem_diff, tw_diff, atol=1.4)
        assert np.isclose(log_flop_diff, tw_diff, atol=1.4)

if __name__ == '__main__':
    test_tamaki_trimming_opt()
