import qtensor
import numpy as np
from qtensor.compression import compressed_contraction_cost
from qtensor.tests import get_test_problem
from qtensor.optimisation import QtreeTensorNet
from qtensor import QtreeQAOAComposer


def costs_to_csv(costs):
    first_line = "flops, memory, width, compressions, decompressions, time"
    lines = [first_line]
    for i, c in enumerate(costs):
        time = c.time(1e6, 1e5, 1e5, 13)
        lines.append(f"[{i}]\t{c.flops},\t{round(c.memory)},\t{c.width},\t {c.compressions},\t{c.decompressions},\t{time}")
    return "\n".join(lines)

def test_compressed_contraction_cost():
    G, gamma, beta = get_test_problem(n=20, p=4, d=4)

    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    tn = QtreeTensorNet.from_qtree_gates(composer.circuit)
    max_time = 15
    opt = qtensor.toolbox.get_ordering_algo('greedy')
    peo, t = opt.optimize(tn)
    print(f"Contraction width: {opt.treewidth}")
    M_limit = opt.treewidth - 6
    # -- Estimate compressed contraction
    costs = compressed_contraction_cost(tn, peo, mem_limit=M_limit)
    cost = sum(costs[2:], costs[0])
    print(costs_to_csv(costs))
    # -- Estimate regular contraction
    mems_lg, flops_lg = tn.simulation_cost(peo)
    ignored_vars = tn.bra_vars + tn.ket_vars
    peo = [x for x in peo if x not in ignored_vars]
    peo = list(map(int, peo))
    nodes, path = qtensor.utils.get_neighbors_path(tn.get_line_graph(), peo)
    print("Path\n", path)
    # -- Estimate sliced contraction
    opt_par  = qtensor.optimisation.SlicesOptimizer(max_tw=M_limit+1, max_slice=5)
    peo, par_vars, tn = opt_par.optimize(tn)
    print("Par vars", par_vars)
    tn.slice({i: slice(0, 1) for i in par_vars})
    peo_sl= peo[:-len(par_vars)]
    costs_sliced = compressed_contraction_cost(tn, peo_sl)
    cost_sliced = sum(costs_sliced[1:], costs_sliced[0])
    runs_count = 2**len(par_vars)
    # print flops and memory from sliced simulation cost
    flops_run = cost_sliced.flops
    mem_run = cost_sliced.memory
    print(cost)
    print(cost_sliced)
    FLOP_perS = 1e9
    Throughput = 1e11
    print(f'Contraction cost (sliced): {np.log2(flops_run*runs_count*1.)} flops, {np.log2(mem_run*1.)} memory, {cost_sliced.width} width')
    print(f'Contraction cost (old): {np.log2(sum(flops_lg))} flops, {np.log2(max(mems_lg))} memory')
    mems_lg, flops_lg = tn.simulation_cost(peo)
    print(f'Sliced contraction cost (old): {np.log2(sum(flops_lg)*runs_count)} flops, {np.log2(max(mems_lg))} memory')

    print(f'-- Compressed Contraction time estimate: {cost.time(FLOP_perS, Throughput, Throughput, M_limit)} seconds')
    print(f'-- Sliced contraction time estimate: {runs_count*cost_sliced.time(FLOP_perS, Throughput, Throughput, M_limit)} seconds')
    print(f'Contraction time (old): {sum(flops_lg)/FLOP_perS} seconds')


    print("Path list comp\n", [c.width for c in costs])
    print("Maxw", max(path))
    assert opt.treewidth == cost.width

if __name__ == '__main__':
    test_compressed_contraction_cost()
