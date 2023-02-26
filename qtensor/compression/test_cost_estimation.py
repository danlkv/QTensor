import qtensor
import numpy as np
from qtensor.compression import compressed_contraction_cost
from qtensor.tests import get_test_problem
from qtensor.optimisation import QtreeTensorNet
from qtensor import QtreeQAOAComposer
from qtensor.optimisation.Optimizer import TreeTrimSplitter


def costs_to_csv(costs):
    first_line = "flops, memory, width, compressions, decompressions, time"
    lines = [first_line]
    for i, c in enumerate(costs):
        time = c.time(1e11/16, 200e9/16, 200e9/15, 13)
        lines.append(f"[{i}]\t{c.flops},\t{round(c.memory)},\t{c.width},\t {c.compressions},\t{c.decompressions},\t{time}")
    return "\n".join(lines)

def test_compressed_contraction_cost():
    G, gamma, beta = get_test_problem(n=32, p=15, d=4)
    opt = qtensor.toolbox.get_ordering_algo('naive')

    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    tn = QtreeTensorNet.from_qtree_gates(composer.circuit)
    #max_time = 15
    peo, t = opt.optimize(tn)
    print(f"Contraction width: {opt.treewidth}")
    M_limit = opt.treewidth-6
    # -- Estimate compressed contraction
    costs = compressed_contraction_cost(tn, peo, mem_limit=M_limit, compression_ratio=64)
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
    opt_par  = qtensor.optimisation.SlicesOptimizer(base_ordering=opt, max_tw=M_limit+1, max_slice=2+opt.treewidth-M_limit)
    opt_par  = TreeTrimSplitter(base_ordering=opt, max_tw=M_limit+1, max_slice=5+opt.treewidth-M_limit)
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
    print("M limit", M_limit)
    print("Cost", cost)
    print("Cost sliced", cost_sliced)
    FLOP_perS = 1e12
    Throughput = 200e9/16
    print(f'Contraction cost (sliced): {np.log2(flops_run*runs_count*1.)} flops, {np.log2(mem_run*1.)} memory, {cost_sliced.width} width')
    print(f'Contraction cost (old): {np.log2(sum(flops_lg)*1.)} flops, {np.log2(max(mems_lg))} memory')
    mems_lg, flops_lg = tn.simulation_cost(peo)
    print(f'Sliced contraction cost (old): {np.log2(sum(flops_lg)*1.0*runs_count)} flops, {np.log2(max(mems_lg)*1.0)} memory')

    print(f'-- Compressed Contraction time estimate: {cost.time(FLOP_perS, Throughput, Throughput, M_limit)} seconds')
    print(f'-- Sliced contraction time estimate: {runs_count*cost_sliced.time(FLOP_perS, Throughput, Throughput, M_limit)} seconds')
    print(f'Contraction time (old): {sum(flops_lg)/FLOP_perS} seconds')


    print("Path list comp\n", [c.width for c in costs])
    print("Maxw", max(path))
    assert opt.treewidth == cost.width

if __name__ == '__main__':
    test_compressed_contraction_cost()
