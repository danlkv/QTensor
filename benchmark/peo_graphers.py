import matplotlib.pyplot as plt
import numpy as np


def compare_tamaki_heuristic_runtimes_treewidths():
    one_problem_graph_size, one_nodes, one_max_treewidth = read_peo_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun1_maxNodes151_d3_p2_operators-diagonal_seed25.txt")
    # two_problem_graph_size, two_nodes, two_max_treewidth = read_peo_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun2_maxNodes151_d3_p2_operators-diagonal_seed25.txt")
    five_problem_graph_size, five_nodes, five_max_treewidth = read_peo_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun5_maxNodes151_d3_p2_operators-diagonal_seed25.txt")
    fifteen_problem_graph_size, fifteen_nodes, fifteen_max_treewidth = read_peo_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun15_maxNodes151_d3_p2_operators-diagonal_seed25.txt")
    sixty_problem_graph_size, sixty_nodes, sixty_max_treewidth = read_peo_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun60_maxNodes151_d3_p2_operators-diagonal_seed25.txt")

    # greedy_problem_graph_size, greedy_nodes, greedy_max_treewidth = read_peo_file("peo_bench_data/peo_greedy_heuristicRun1_maxNodes501_d3_p2_operators-diagonal_seed25.txt")

    plt.figure()

    plt.title("Heuristic tamaki peo solver for D=3, P=2 QAOA")
    plt.xlabel('Problem graph size')
    plt.ylabel('Tensor network reewidth')
    plt.plot(one_problem_graph_size, one_max_treewidth, 'ro', #two_problem_graph_size, two_max_treewidth, 'os',
             five_problem_graph_size, five_max_treewidth, 'gs', fifteen_problem_graph_size, fifteen_max_treewidth, 'yd',
             sixty_problem_graph_size, sixty_max_treewidth, 'bP')
    plt.legend(["1 sec", "5 sec", "15 sec", "60 sec"])

    plt.show()


def compare_diagonal_to_full_matrix_operators():
    diag_problem_graph_size, diag_nodes, diag_max_treewidth = read_peo_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun15_maxNodes501_d3_p2_operators-diagonal_seed25.txt")
    full_problem_graph_size, full_nodes, full_max_treewidth = read_peo_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun15_maxNodes501_d3_p2_operators-full_matrix_seed25.txt")

    plt.figure()

    plt.title("Heuristic tamaki peo solver for D=3, P=2 QAOA using different operators")
    plt.xlabel('Problem graph size')
    plt.ylabel('Treewidth')
    plt.plot(diag_problem_graph_size, diag_max_treewidth, 'bo', full_problem_graph_size, full_max_treewidth, 'rs')
    plt.legend(["diagonal", "full matrix"])

    plt.show()


def tw_reliance_p():
    """
    To graph whether treewidth depends on QAOA p value
    """
    d_values = [3, 4, 5]
    d_3_max_treewidth = []
    d_4_max_treewidth = []
    d_5_max_treewidth = []

    p_values = np.arange(1, 11)

    for p_i in p_values:
        _, _, tw = read_peo_file("treewidth_dependency_data/peo_tamaki_heuristic_heuristicRun120_maxNodes21_d3_p" + str(p_i) + "_operators-diagonal_seed25.txt")
        d_3_max_treewidth.append(tw)

    for p_i in p_values:
        _, _, tw = read_peo_file("treewidth_dependency_data/peo_tamaki_heuristic_heuristicRun120_maxNodes21_d4_p" + str(p_i) + "_operators-diagonal_seed25.txt")
        d_4_max_treewidth.append(tw)

    for p_i in p_values:
        _, _, tw = read_peo_file("treewidth_dependency_data/peo_tamaki_heuristic_heuristicRun120_maxNodes21_d5_p" + str(p_i) + "_operators-diagonal_seed25.txt")
        d_5_max_treewidth.append(tw)

    plt.figure()

    plt.title("P val vs Treewidth on 20 node problems")
    plt.xlabel('QAOA P Value')
    plt.ylabel('Tensor Network Treewidth')
    plt.plot(p_values, d_3_max_treewidth, 'bo--', p_values, d_4_max_treewidth, 'rs--', d_5_max_treewidth, 'gh--')
    plt.legend(["d = 3", "d = 4", "d = 5"])

    plt.show()


def tw_reliance_connectivity():
    """
    To graph whether treewidth depends on problem graph connectivity
    """
    p_values = [1, 2, 3]
    p_1_max_treewidth = []
    p_2_max_treewidth = []
    p_3_max_treewidth = []

    d_values = np.arange(3, 6)

    for d_i in d_values:
        _, _, tw = read_peo_file(
            "treewidth_dependency_data/peo_tamaki_heuristic_heuristicRun120_maxNodes21_d" + str(d_i) + "_p1_operators-diagonal_seed25.txt")
        p_1_max_treewidth.append(*tw)

    for d_i in d_values:
        _, _, tw = read_peo_file(
            "treewidth_dependency_data/peo_tamaki_heuristic_heuristicRun120_maxNodes21_d" + str(d_i) + "_p2_operators-diagonal_seed25.txt")
        p_2_max_treewidth.append(*tw)

    for d_i in d_values:
        _, _, tw = read_peo_file(
            "treewidth_dependency_data/peo_tamaki_heuristic_heuristicRun120_maxNodes21_d" + str(d_i) + "_p3_operators-diagonal_seed25.txt")
        p_3_max_treewidth.append(*tw)

    plt.figure()

    plt.title("Graph connectivity vs Treewidth on 20 node problems")
    plt.xlabel('Problem graph connectivity')
    plt.ylabel('Tensorr Network Treewidth')
    plt.plot(d_values, p_1_max_treewidth, 'bo--', d_values, p_2_max_treewidth, 'rs--', d_values, p_3_max_treewidth, 'gh--')
    plt.legend(["p = 1", "p = 2", "p = 3"])

    plt.show()



def read_peo_file(filename):
    problem_graph_size=[]
    nodes = []
    max_treewidth = []

    with open(filename, "r") as file:
        for line in file:
            if line == "\n":
                continue
            split_line = line.split(" ")
            key = split_line[0]
            val = split_line[1]

            if key == "problem_graph_size:":
                problem_graph_size.append(int(val))
            elif key == "graph_nodes:":
                nodes.append(int(val))
            elif key == "max_treewidth:":
                max_treewidth.append(int(val))

    return problem_graph_size, nodes, max_treewidth


compare_tamaki_heuristic_runtimes_treewidths()
# compare_diagonal_to_full_matrix_operators()
# tw_reliance_connectivity()
# tw_reliance_p()

