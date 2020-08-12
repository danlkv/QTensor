import matplotlib.pyplot as plt
import numpy as np
from statistics import median

def get_min_max_median_per_index(zipped_list):
    """
    Parameters
    ----------
    zipped_list: All list values zipped together. Ideally at least 3

    Returns
    -------
    min_max_median_list: To be used with matplotlib fill_between. Min is the bottom, Max is the top, median
        is another line in the middle
    """
    min_list = []
    max_list = []
    median_list = []

    for item_i in zipped_list:
        min_list.append(min(item_i))
        max_list.append(max(item_i))
        median_list.append(median(item_i))

    return min_list, max_list, median_list


def compare_tamaki_heuristic_runtimes_treewidths():
    one_problem_graph_size, one_nodes, one_max_treewidth = read_text_data_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun1_maxNodes151_d3_p2_operators-diagonal_seed25.txt")
    # two_problem_graph_size, two_nodes, two_max_treewidth = read_peo_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun2_maxNodes151_d3_p2_operators-diagonal_seed25.txt")
    five_problem_graph_size, five_nodes, five_max_treewidth = read_text_data_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun5_maxNodes151_d3_p2_operators-diagonal_seed25.txt")
    fifteen_problem_graph_size, fifteen_nodes, fifteen_max_treewidth = read_text_data_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun15_maxNodes151_d3_p2_operators-diagonal_seed25.txt")
    sixty_problem_graph_size, sixty_nodes, sixty_max_treewidth = read_text_data_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun60_maxNodes151_d3_p2_operators-diagonal_seed25.txt")

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


def compare_tamaki_to_flow_cutter():
    seeds = [23, 24]
    methods = ["peo_tamaki_heuristic", "peo_flow_cutter"]
    d_value = 3
    p_value = 2
    nodes = np.arange(10, 101, 10)
    runtime = 1800
    plot_color = ["b", "c", "g"]
    plot_color_shapes = ["bo-", "cP-", "gv-"]

    method_values_lists = {}
    # for each d_val, min, max, and median treewidth
    for method in methods:
        method_values_lists[method] = [[], [], []]

    for method in methods:
        seed_lists = []
        for seed_i in seeds:
            _, _, one_seed_list = read_text_data_file(
                "peo_bench_data/" + method + "_heuristicRun" + str(runtime) + "_maxNodes" + str(max(nodes)+1)
                + "_d" + str(d_value) + "_p" + str(p_value) + "_operators-diagonal_seed" + str(seed_i) + ".txt")
            seed_lists.append(one_seed_list)
        zipped_lists = zip(*seed_lists)
        method_values_lists[method] = get_min_max_median_per_index(zipped_lists)  # place all three lists into d_lists

    plt.figure()
    plt.title("Problem Nodes vs Treewidth on D=3, P=2 Random Graphs")
    plt.xlabel("Problem Graph Nodes")
    plt.ylabel('Tensor Network Treewidth')

    for i, method in enumerate(methods):
        plt.plot(nodes, method_values_lists[method][2], plot_color_shapes[i])
        plt.fill_between(nodes, method_values_lists[method][0], method_values_lists[method][1], color=plot_color[i], alpha=0.2)

    legend = []
    for method in methods:
        legend.append(str(method))
    plt.legend(legend)

    plt.show()

def compare_diagonal_to_full_matrix_operators():
    diag_problem_graph_size, diag_nodes, diag_max_treewidth = read_text_data_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun15_maxNodes501_d3_p2_operators-diagonal_seed25.txt")
    full_problem_graph_size, full_nodes, full_max_treewidth = read_text_data_file("peo_bench_data/peo_tamaki_heuristic_heuristicRun15_maxNodes501_d3_p2_operators-full_matrix_seed25.txt")

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
    p_values = np.arange(1, 9)
    seeds = [23]
    method = "peo_tamaki_heuristic"
    runtime = 1800
    nodes = 37
    d_values_lists = {}
    plot_color = ["b", "c", "g"]
    plot_color_shapes = ["bo-", "cP-", "gv-"]
    # for each d_val, min, max, and median treewidth
    for d_i in d_values:
        d_values_lists[d_i] = [[], [], []]

    for d_i in d_values:
        seed_lists = []
        for seed_i in seeds:
            one_seed_list = []
            for p_i in p_values:
                _, _, tw = read_text_data_file(
                    "treewidth_dependency_data/" + method + "_heuristicRun" + str(runtime) + "_maxNodes" + str(nodes)
                    + "_d" + str(d_i) + "_p" + str(p_i) + "_operators-diagonal_seed" + str(seed_i) + ".txt")
                one_seed_list.append(*tw)
            seed_lists.append(one_seed_list)
        zipped_lists = zip(*seed_lists)
        d_values_lists[d_i] = get_min_max_median_per_index(zipped_lists)  # place all three lists into d_lists

    plt.figure()
    plt.title("P val vs Treewidth on 36 node problems")
    plt.xlabel('QAOA P Value')
    plt.ylabel('Tensor Network Treewidth')

    for i, d_i in enumerate(d_values):
        plt.plot(p_values, d_values_lists[d_i][2], plot_color_shapes[i])
        plt.fill_between(p_values, d_values_lists[d_i][0], d_values_lists[d_i][1], color=plot_color[i], alpha=0.2)

    legend = []
    for d_i in d_values:
        legend.append("d = " + str(d_i))
    plt.legend(legend)

    plt.show()


def tw_reliance_connectivity():
    """
    To graph whether treewidth depends on problem graph connectivity
    """
    p_values = [1, 2, 3]
    d_values = np.arange(3, 6)
    seeds = [23, 24]
    method = "peo_tamaki_heuristic"
    runtime = 1800
    nodes = 37
    p_values_lists = {}
    plot_color = ["b", "c", "g"]
    plot_color_shapes = ["bo-", "cP-", "gv-"]
    # for each p_val, min, max, and median treewidth
    for p_i in p_values:
        p_values_lists[p_i] = [[], [], []]

    for p_i in p_values:
        seed_lists = []
        for seed_i in seeds:
            one_seed_list = []
            for d_i in d_values:
                _, _, tw = read_text_data_file(
                    "treewidth_dependency_data/" + method + "_heuristicRun" + str(runtime) + "_maxNodes" + str(nodes)
                    + "_d" + str(d_i) + "_p" + str(p_i) + "_operators-diagonal_seed" + str(seed_i) + ".txt")
                one_seed_list.append(*tw)
            seed_lists.append(one_seed_list)
        zipped_lists = zip(*seed_lists)
        p_values_lists[p_i] = get_min_max_median_per_index(zipped_lists)  # place all three lists into d_lists

    plt.figure()
    plt.title("Connectivity vs Treewidth on 36 node problems")
    plt.xlabel("Problem Graph Connectivity")
    plt.ylabel('Tensor Network Treewidth')
    plt.xticks(d_values)

    for i, p_i in enumerate(p_values):
        plt.plot(d_values, p_values_lists[p_i][2], plot_color_shapes[i])
        plt.fill_between(d_values, p_values_lists[p_i][0], p_values_lists[p_i][1], color=plot_color[i], alpha=0.2)

    legend = []
    for p_i in p_values:
        legend.append("p = " + str(p_i))
    plt.legend(legend)

    plt.show()


def read_text_data_file(filename):
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


# compare_tamaki_heuristic_runtimes_treewidths()
# compare_diagonal_to_full_matrix_operators()
# compare_tamaki_to_flow_cutter()
# tw_reliance_connectivity()
tw_reliance_p()


