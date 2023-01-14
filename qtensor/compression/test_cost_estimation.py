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
    M_limit = opt.treewidth + 10
    costs = compressed_contraction_cost(tn, peo, mem_limit=M_limit)
    cost = sum(costs[1:], costs[0])
    print(costs_to_csv(costs))
    print(cost)
    print(f'Contraction time estimate: {cost.time(1e6, 1e5, 1e5, M_limit)} seconds')
    mems_lg, flops_lg = tn.simulation_cost(peo)
    print(f'Contraction cost (old): {np.log2(sum(flops_lg))} flops, {np.log2(max(mems_lg))} memory')
    print(f'Contraction time (old): {sum(flops_lg)/1e6} seconds')
    ignored_vars = tn.bra_vars + tn.ket_vars
    peo = [x for x in peo if x not in ignored_vars]
    peo = list(map(int, peo))
    nodes, path = qtensor.utils.get_neighbors_path(tn.get_line_graph(), peo)
    print("Path\n", path)
    print("Path list comp\n", [c.width for c in costs])
    print("Maxw", max(path))
    assert opt.treewidth == cost.width

if __name__ == '__main__':
    test_compressed_contraction_cost()
