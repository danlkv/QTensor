"""
Create different benchmarks for running peo solvers with
multiple options and types of solvers
"""

import sys
sys.path.insert(0, "../")
sys.path.insert(1, "../../PACE2017-TrackA")
sys.path.insert(2, "../../quickbb")
sys.path.insert(3, "../qtree")
sys.path.insert(4, "../../flow-cutter-pace17")

import numpy as np
import networkx as nx
from qensor import QtreeQAOAComposer, CirqQAOAComposer
import qtree
from qtree.graph_model import get_upper_bound_peo, get_peo
from time import time
from qensor import utils
from convert_data_to_mongodb import print_mongodb_info


def create_qaoa_circuit_from_problem_graph(num_nodes, graph_connectivity, seed, beta, gamma, operators):
    graph = nx.random_regular_graph(d=graph_connectivity, n=num_nodes, seed=seed)
    composer = QtreeQAOAComposer(graph, beta=beta, gamma=gamma)
    # for using the non-optimal full operators
    if operators == "full_matrix":
        composer.set_operators(operators)
    composer.ansatz_state()  # creates the QAOA circuit

    # create the tensor network graph based on the problem circuit
    return composer.circuit


def peo_benchmark(
        method,
        problem_graph_start,
        problem_graph_end,
        problem_graph_jumps=2,
        graph_connectivity=5,
        heuristic_runtime=-1,
        operators="diagonal",
        seed=25,
        beta=None,
        gamma=None):
    if beta is None:
        beta = [.5, 1]
    if gamma is None:
        gamma = [.5, 1]

    for num_nodes in range(problem_graph_start, problem_graph_end, problem_graph_jumps):
        # print("problem_graph_size:", num_nodes)
        qc = create_qaoa_circuit_from_problem_graph(num_nodes, graph_connectivity, seed, beta, gamma, operators)

        all_gates = qc
        n_qubits = len(set(sum([g.qubits for g in all_gates], tuple())))
        circuit = [[g] for g in qc]
        buckets, data_dict, bra_vars, ket_vars = qtree.optimizer.circ2buckets(
            n_qubits, circuit)
        graph = qtree.graph_model.buckets2graph(buckets, ignore_variables=ket_vars + bra_vars)

        # find the perfect elimination order (peo) for the tensor network
        peo = []
        treewidth = -1
        # time the peo functions
        start = time()

        if method == "tamaki_heuristic":
            peo, treewidth = get_upper_bound_peo(graph, method="tamaki", wait_time=heuristic_runtime)

        elif method == "quickbb":
            peo, treewidth = get_upper_bound_peo(graph, method="quickbb", wait_time=heuristic_runtime)

        elif method == "flow_cutter":
            peo, treewidth = get_upper_bound_peo(graph, method="flow_cutter", wait_time=heuristic_runtime)

        elif method == "greedy":
            peo_ints, tw = utils.get_locale_peo(graph, utils.n_neighbors)
            treewidth = max(tw)
            peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                                       name=graph.nodes[var]['name'])
                   for var in peo_ints]

        elapsed = time() - start

        # print all information in a mongodb-friendly format
        print_mongodb_info(method, str(num_nodes), str(graph_connectivity), str(len(beta)),
                           str(heuristic_runtime), operators, str(seed), str(graph.number_of_nodes()), str(treewidth))


def peo_benchmark_wrapper(
        folder,
        method,
        problem_graph_start,
        problem_graph_end,
        problem_graph_jumps=2,
        graph_connectivity=5,
        heuristic_runtime=-1,
        operators="diagonal",
        seed=25,
        beta=None,
        gamma=None):
    if beta is None:
        beta = [.5, 1]
    if gamma is None:
        gamma = [.5, 1]

    # sys.stdout = open(folder + "peo_" + str(method)
    #                   + "_heuristicRun" + str(heuristic_runtime)
    #                   + "_maxNodes" + str(problem_graph_end)
    #                   + "_d" + str(graph_connectivity)
    #                   + "_p" + str(len(beta))
    #                   + "_operators-" + operators
    #                   + "_seed" + str(seed)
    #                   + ".txt", 'w')

    peo_benchmark(method,
                  problem_graph_start,
                  problem_graph_end,
                  problem_graph_jumps,
                  graph_connectivity,
                  heuristic_runtime,
                  operators,
                  seed,
                  beta,
                  gamma)


def cotengra_test():
    sys.path.insert(5, "../../cotengra")
    import quimb.tensor as qtn
    import cotengra as ctg
    import math

    num_nodes = 30
    graph_connectivity = 3
    seed = 24
    beta = [0.5, 0.5]
    gamma = beta
    operators = "diagonal"

    qc = create_qaoa_circuit_from_problem_graph(num_nodes, graph_connectivity, seed, beta, gamma, operators)

    ########
    # Create circuit in the quimb representation
    ########
    gates_to_apply = []
    # for layer in qc:
    for gate in qc:
        if gate.name == "ZPhase":
            gates_to_apply.append(["RZ", *gate.qubits, gate.parameters["alpha"]])
        else:
            gates_to_apply.append([gate.name, *gate.qubits])

    quimb = qtn.Circuit(num_nodes)
    quimb.apply_gates(gates_to_apply)

    print(quimb)

    circ = quimb
    psi_f = qtn.MPS_computational_state('0' * (circ.N))
    tn = quimb.psi & psi_f

    output_inds = []
    # tn.full_simplify_(output_inds=output_inds)
    tn.astype_('complex64')

    opt = ctg.HyperOptimizer(
        # methods=['kahypar', 'greedy', 'walktrap'],
        methods=['kahypar'],
        max_repeats=128,
        progbar=True,
        minimize='size',
        score_compression=0.  # deliberately make the optimizer try many methods
    )

    info = tn.contract(all, optimize=opt, get='path-info')
    print(math.log2(info.largest_intermediate))


def run_treewidth_dependency_benchmarks(seeds):
    p_values = np.ones(100) * 0.5
    # seeds = [23]

    for seed in seeds:
        # run increasing d for several p
        d = np.arange(2, 9)
        for d_i in d:
            peo_benchmark_wrapper("treewidth_dependency_data/", "greedy", 36, 37, 1, d_i, 1800, "diagonal", seed,
                                  p_values[:1], p_values[:1])
            peo_benchmark_wrapper("treewidth_dependency_data/", "greedy", 36, 37, 1, d_i, 1800, "diagonal", seed,
                                  p_values[:2], p_values[:2])
            peo_benchmark_wrapper("treewidth_dependency_data/", "greedy", 36, 37, 1, d_i, 1800, "diagonal", seed,
                                  p_values[:3], p_values[:3])

        # run increasing p for several d
        p_options = np.arange(1, 9)
        for i in p_options:
            peo_benchmark_wrapper("treewidth_dependency_data/", "tamaki_heuristic", 36, 37, 1, 3, 1800, "diagonal", seed,
                                  p_values[:i], p_values[:i])
            peo_benchmark_wrapper("treewidth_dependency_data/", "tamaki_heuristic", 36, 37, 1, 4, 1800, "diagonal", seed,
                                  p_values[:i], p_values[:i])
            peo_benchmark_wrapper("treewidth_dependency_data/", "tamaki_heuristic", 36, 37, 1, 5, 1800, "diagonal", seed,
                                  p_values[:i], p_values[:i])


def run_peo_benchmarks(seeds):
    # peo_benchmark_wrapper("peo_bench_data/", "greedy", 10, 151, 10, 3, 1)
    peo_running_times = [1800] #,1, 5, 15, 60]
    # seeds = [23]
    methods = ["tamaki_heuristic", "flow_cutter"]

    for method in methods:
        for seed in seeds:
            for peo_run_time in peo_running_times:
                peo_benchmark_wrapper("peo_bench_data/", method, 10, 101, 10, 3, peo_run_time, "diagonal", seed)


    # peo_benchmark_wrapper("peo_bench_data/", "tamaki_heuristic", 10, 151, 10, 3, 1, "full_matrix")
    # peo_benchmark_wrapper("peo_bench_data/", "tamaki_heuristic", 10, 151, 10, 3, 2, "full_matrix")
    # peo_benchmark_wrapper("peo_bench_data/", "tamaki_heuristic", 10, 151, 10, 3, 5, "full_matrix")
    # peo_benchmark_wrapper("peo_bench_data/", "tamaki_heuristic", 10, 151, 10, 3, 15, "full_matrix")
    # peo_benchmark_wrapper("peo_bench_data/", "tamaki_heuristic", 10, 151, 10, 3, 60, "full_matrix")

    # peo_benchmark_wrapper("peo_bench_data/", "quickbb", 10, 31, 10, 3, 1)
    # peo_benchmark_wrapper("peo_bench_data/", "quickbb", 10, 31, 10, 3, 2)
    # peo_benchmark_wrapper("peo_bench_data/", "quickbb", 10, 31, 10, 3, 5)
    # peo_benchmark_wrapper("peo_bench_data/", "quickbb", 10, 31, 10, 3, 15)



# peo_benchmark_wrapper("peo_bench_data/", "tamaki_heuristic", 20, 21, 10, 3, 500, "diagonal", 25, [.5, .5, .5, .5], [.5, .5, .5, .5])

# peo_benchmark_wrapper("peo_bench_data/", "greedy", 10, 501, 10, 3, 1)
# test_seeds = [23, 24, 25]
# run_treewidth_dependency_benchmarks(test_seeds)
# run_peo_benchmarks(test_seeds)
cotengra_test()
