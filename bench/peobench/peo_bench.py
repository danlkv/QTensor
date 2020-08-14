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
from qensor import QtreeQAOAComposer
import qtree
from qtree.graph_model import get_upper_bound_peo
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
        num_nodes,
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
        method,
        problem_num_nodes,
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

    """
    Options:
    method: choose from supported methods "greedy", "tamaki_exact", "tamaki_heuristic", "flow_cutter", "quickbb"
    problem_num_nodes: number of nodes in problem graph
    graph_connectivity: problem graph connectivity
    heuristic_runtime: amount of time given to the heuristic solver
    operators: choose from tensor types "diagonal", "full_matrix"
    seed: for the random regular graph generation
    beta: array of beta values for the qaoa problem
    gamma: array of gamma values for the qaoa problem
    """

    peo_benchmark(method,
                  problem_num_nodes,
                  graph_connectivity,
                  heuristic_runtime,
                  operators,
                  seed,
                  beta,
                  gamma)


def kahypar_test():
    import kahypar

    num_nodes = 30
    graph_connectivity = 3
    seed = 24
    beta = [0.5, 0.5]
    gamma = beta
    operators = "diagonal"

    qc = create_qaoa_circuit_from_problem_graph(num_nodes, graph_connectivity, seed, beta, gamma, operators)
    all_gates = qc
    n_qubits = len(set(sum([g.qubits for g in all_gates], tuple())))
    circuit = [[g] for g in qc]
    buckets, data_dict, bra_vars, ket_vars = qtree.optimizer.circ2buckets(
        n_qubits, circuit)
    hypergraph = qtree.graph_model.buckets2hypergraph(buckets)

    context = kahypar.Context()
    context.loadINIconfiguration("/Users/filipmazurek/Documents/Simulator_Argonne/kahypar/config/cut_kKaHyPar_sea20.ini")

    k = 2

    context.setK(k)
    context.setEpsilon(0.03)

    kahypar.partition(hypergraph, context)


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
    p_options = np.arange(1, 9)
    d = np.arange(2, 9)
    problem_graph_nodes = 36
    method = "tamaki_heuristic"
    heuristic_method_runtime = 1800  # in seconds

    for seed in seeds:
        # run increasing d for several set p
        for d_i in d:
            peo_benchmark_wrapper(method, problem_graph_nodes, d_i, heuristic_method_runtime, "diagonal", seed,
                                  p_values[:1], p_values[:1])
            peo_benchmark_wrapper(method, problem_graph_nodes, d_i, heuristic_method_runtime, "diagonal", seed,
                                  p_values[:2], p_values[:2])
            peo_benchmark_wrapper(method, problem_graph_nodes, d_i, heuristic_method_runtime, "diagonal", seed,
                                  p_values[:3], p_values[:3])

        # run increasing p for several set d
        for i in p_options:
            peo_benchmark_wrapper(method, problem_graph_nodes, 3, heuristic_method_runtime, "diagonal", seed,
                                  p_values[:i], p_values[:i])
            peo_benchmark_wrapper(method, problem_graph_nodes, 4, heuristic_method_runtime, "diagonal", seed,
                                  p_values[:i], p_values[:i])
            peo_benchmark_wrapper(method, problem_graph_nodes, 5, heuristic_method_runtime, "diagonal", seed,
                                  p_values[:i], p_values[:i])


def run_peo_benchmarks(seeds):
    peo_running_times = [1800]
    methods = ["tamaki_heuristic", "flow_cutter"]
    problem_nodes_list = np.arange(10, 151, 10)
    problem_graph_connectivities = [3]

    for method in methods:
        for seed in seeds:
            for peo_run_time in peo_running_times:
                for problem_nodes in problem_nodes_list:
                    for problem_graph_connectivity in problem_graph_connectivities:
                        peo_benchmark_wrapper(method, problem_nodes, problem_graph_connectivity, peo_run_time, "diagonal", seed)


"""
The problem graph seed is given as a problem parameter 
"""
# run_treewidth_dependency_benchmarks(test_seeds)
# run_peo_benchmarks(test_seeds)
# cotengra_test()
# kahypar_test()
