import numpy as np


def read_text_data_file():
    folders = ["treewidth_dependency_data", "peo_bench_data"]
    methods = ["flow_cutter", "tamaki_heuristic", "greedy"]
    runtimes = [1, 2, 5, 15, 60, 120, 1800]
    max_nodes_list = [11, 21, 31, 36, 37, 51, 101, 151, 501]
    d_list = np.arange(2, 9)
    p_list = np.arange(1, 9)
    operators = ["diagonal", "full_matrix"]
    seeds = [23, 24, 25]

    for folder in folders:
        for method in methods:
            for runtime in runtimes:
                for max_nodes in max_nodes_list:
                    for d_i in d_list:
                        for p_i in p_list:
                            for operator in operators:
                                for seed in seeds:
                                    if method == "greedy":
                                        runtime = -1

                                    identifier = folder + "/peo_" + method + "_heuristicRun" + str(runtime) + "_maxNodes" + str(max_nodes) + "_d" + str(d_i) + "_p" + str(p_i) + "_operators-" + operator + "_seed" + str(seed)
                                    extension = ".txt"
                                    filename = identifier + extension

                                    problem_graph_size, nodes, max_treewidth = -1, -1, -1

                                    try:
                                        with open(filename, "r") as file:
                                            for line in file:
                                                if line == "\n":
                                                    if problem_graph_size > 0 and nodes > 0 and max_treewidth > 0:
                                                        print_mongodb_info(method, problem_graph_size, d_i, p_i, runtime, operator, seed, nodes, max_treewidth)
                                                        problem_graph_size, nodes, max_treewidth = -1, -1, -1
                                                    continue
                                                split_line = line.split(" ")
                                                key = split_line[0]
                                                val = split_line[1]

                                                if key == "problem_graph_size:":
                                                    problem_graph_size = int(val)
                                                elif key == "graph_nodes:":
                                                    nodes = int(val)
                                                elif key == "max_treewidth:":
                                                    max_treewidth = int(val)
                                    except FileNotFoundError:
                                        continue


def print_mongodb_info(method, problem_graph_size, d_i, p_i, runtime, operator, seed, nodes, treewidth):
    if method == "greedy":
        id_string = "method" + method + "_problemSize" + str(problem_graph_size) + "_problemConnectivity" + str(d_i) + \
                    "_qaoaP" + str(p_i) + "_operators" + operator + \
                    "_seed" + str(seed) + "_graphNodes" + str(nodes) + "_treewidth" + str(treewidth)
        print("{'_id': '" + id_string + "', 'method': '" + method +
              ", 'problem_args: " +
              "{'problem_graph_size': " + str(problem_graph_size) +
              ", 'problem_graph_connectivity': " + str(d_i) + ", 'qaoa_p_value': " + str(p_i) +
              "', 'seed': " + str(seed) +
              "}" +
              ", 'operators': '" + operator + ", 'graph_nodes': " + str(nodes) +
              ", 'treewidth': " + str(treewidth) + "}")

    elif method == "flow_cutter" or method == "tamaki_heuristic":
        id_string = "method" + method + "_problemSize" + str(problem_graph_size) + "_problemConnectivity" + str(d_i) + \
                    "_qaoaP" + str(p_i) + "_heuristic_method_runtime" + str(runtime) + "_operators" + operator + \
                    "_seed" + str(seed) + "_graphNodes" + str(nodes) + "_treewidth" + str(treewidth)
        print("{'_id': '" + id_string + "', 'method': '" + method +
              ", 'method_args: {'heuristic_method_runtime': " + str(runtime) + "}" +
              ", 'problem_args: " +
              "{'problem_graph_size': " + str(problem_graph_size) +
              ", 'problem_graph_connectivity': " + str(d_i) + ", 'qaoa_p_value': " + str(p_i) +
              "', 'seed': " + str(seed) +
              "}" +
              ", 'operators': '" + operator + ", 'graph_nodes': " + str(nodes) +
              ", 'treewidth': " + str(treewidth) + "}")


read_text_data_file()
