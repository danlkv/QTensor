"""
Creates benchmarks for using the qtree backend.
The qtree backend may use either full matrix operators or a diagonal shorthand.
This benchmark checks for differences in using the diagonal speedup
"""
import sys
import os.path
import numpy as np
import networkx as nx
from qensor import QtreeQAOAComposer
from qensor.Simulate import QtreeSimulator
from qensor.ProcessingFrameworks import PerfNumpyBackend
import matplotlib.pyplot as plt

sys.path.insert(0, "../")
sys.path.insert(1, "../../PACE2017-TrackA")


def tamaki_backend_bench():
    for num_nodes in range(6, 35, 2):
        print("problem_graph_size:", num_nodes)
        graph = nx.random_regular_graph(d=5, n=num_nodes, seed=25)
        composer = QtreeQAOAComposer(graph, beta=[1,.5], gamma=[1,.5])
        composer.ansatz_state()
        performance_backend = PerfNumpyBackend()
        sim = QtreeSimulator(bucket_backend=performance_backend)
        result = sim.simulate(composer.circuit)
        # print(result.data)
        print(performance_backend.gen_report())
        # print("treewidth: ", result)


# def performance_backend_simulation():
#     for num_nodes in range(6, 27, 2):
#         print("problem_graph_size:", num_nodes)
#         graph = nx.random_regular_graph(d=5, n=num_nodes, seed=25)
#         composer = QtreeQAOAComposer(graph, beta=[1,.5], gamma=[1,.5])
#         composer.ansatz_state()
#         performance_backend = PerfNumpyBackend()
#         sim = QtreeSimulator(bucket_backend=performance_backend)
#         result = sim.simulate(composer.circuit)
#         # print(result.data)
#         print(performance_backend.gen_report())
#         # print("treewidth: ", result)


def plot_simulation_results():
    tamaki_problem_graph_size = []
    tamaki_nodes = []
    tamaki_max_treewidth = []
    tamaki_num_bucket_contractions = []
    tamaki_total_time = []
    tamaki_peo = []
    greedy_problem_graph_size = []
    greedy_nodes = []
    greedy_max_treewidth = []
    greedy_num_bucket_contractions = []
    greedy_total_time = []
    greedy_peo = []
    with open("bench_tamaki_data/tamaki_heuristic_d5_p2.txt", "r") as file:
        for line in file:
            if line == "\n":
                continue
            split_line = line.split(" ")
            key = split_line[0]
            val = split_line[1]
            # print(key)
            # print(val)
            if key == "problem_graph_size:":
                tamaki_problem_graph_size.append(int(val))
            if key == "graph_nodes:":
                tamaki_nodes.append(int(val))
            if key == "max_treewidth:":
                tamaki_max_treewidth.append(int(val))
            if key == "Total_time:":
                tamaki_total_time.append(float(val))
            if key == "Num_bucket_contractions:":
                tamaki_num_bucket_contractions.append(float(val))
            if key == "peo_processing:":
                tamaki_peo.append(float(val))


    with open("bench_tamaki_data/flow_cutter_d5_p2.txt", "r") as file:
        for line in file:
            if line == "\n":
                continue
            split_line = line.split(" ")
            key = split_line[0]
            val = split_line[1]
            # print(key)
            # print(val)
            if key == "problem_graph_size:":
                greedy_problem_graph_size.append(int(val))
            if key == "graph_nodes:":
                greedy_nodes.append(int(val))
            if key == "max_treewidth:":
                greedy_max_treewidth.append(int(val))
            if key == "Total_time:":
                greedy_total_time.append(float(val))
            if key == "Num_bucket_contractions:":
                greedy_num_bucket_contractions.append(float(val))
            if key == "peo_processing:":
                greedy_peo.append(float(val))

    # plt.plot(problem_graph_size, diag_nodes, 'r--', problem_graph_size, full_nodes, 'bs')

    plt.figure()

    # plt.subplot(211)
    # plt.title("Peo solvers for D=5, P=2 QAOA")
    # plt.xlabel('Problem graph size')
    # plt.ylabel('Peo runtime (s)')
    # plt.plot(tamaki_problem_graph_size, tamaki_peo, 'bo', greedy_problem_graph_size, greedy_peo, 'rs')
    # plt.xticks(np.arange(2, 27, 2))
    # plt.legend(["tamaki", "greedy"])

    plt.subplot(211)
    plt.title("Peo solvers for D=5, P=2 QAOA")
    plt.xlabel('Problem graph size')
    plt.ylabel('treewidth')
    plt.plot(tamaki_problem_graph_size, tamaki_max_treewidth, 'bo', greedy_problem_graph_size[:17], greedy_max_treewidth[:17], 'rs')
    plt.xticks(np.arange(2, 35, 2))
    # plt.yticks(np.arange(0, 17, 2))
    plt.legend(["heurisitic tamaki", "flow cutter"])

    # plt.subplot(211)
    # plt.title("Peo solvers for D=5, P=2 QAOA")
    # plt.xlabel('Problem graph size')
    # plt.ylabel('Peo runtime (s)')
    # plt.plot(tamaki_problem_graph_size, tamaki_peo, 'bo', greedy_problem_graph_size, greedy_peo, 'rs')
    # plt.xticks(np.arange(2, 27, 2))
    # plt.legend(["tamaki", "greedy"])
    #
    plt.subplot(212)
    plt.xlabel('Problem graph size')
    plt.ylabel('contr'
               'action runtime')
    plt.plot(tamaki_problem_graph_size, tamaki_total_time, 'bo', greedy_problem_graph_size[:17], greedy_total_time[:17],
             'rs')
    plt.xticks(np.arange(2, 35, 2))
    # plt.yticks(np.arange(0, 17, 2))
    plt.legend(["heurisitic tamaki", "flow cutter"
                                     ""])

    # plt.plot(diag_problem_graph_size, diag_nodes, 'bo', full_problem_graph_size, full_nodes, 'rs')
    plt.show()


plot_simulation_results()
# tamaki_backend_bench()