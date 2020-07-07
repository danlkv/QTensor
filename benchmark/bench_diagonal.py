"""
Creates benchmarks for using the qtree backend.
The qtree backend may use either full matrix operators or a diagonal shorthand.
This benchmark checks for differences in using the diagonal speedup
"""
import sys
import os.path
sys.path.insert(0, "../")
sys.path.insert(1, "../../PACE2017-TrackA")

import numpy as np
import networkx as nx
from qensor import QtreeQAOAComposer
from qensor.Simulate import QtreeSimulator
from qensor.ProcessingFrameworks import PerfNumpyBackend
import matplotlib.pyplot as plt


def performance_backend_simulation():
    for num_nodes in range(6, 27, 2):
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


def plot_simulation_results():
    diag_problem_graph_size = []
    diag_nodes = []
    diag_max_treewidth = []
    diag_num_bucket_contractions = []
    diag_total_time = []
    full_problem_graph_size = []
    full_nodes = []
    full_max_treewidth = []
    full_num_bucket_contractions = []
    full_total_time = []
    with open("bench_diagonal_data/diag_bench.txt", "r") as file:
        for line in file:
            if line == "\n":
                continue
            split_line = line.split(" ")
            key = split_line[0]
            val = split_line[1]
            # print(key)
            # print(val)
            if key == "problem_graph_size:":
                diag_problem_graph_size.append(int(val))
            if key == "graph_nodes:":
                diag_nodes.append(int(val))
            if key == "max_treewidth:":
                diag_max_treewidth.append(int(val))
            if key == "Total_time:":
                diag_total_time.append(float(val))
            if key == "Num_bucket_contractions:":
                diag_num_bucket_contractions.append(float(val))

    with open("bench_diagonal_data/full_bench.txt", "r") as file:
        for line in file:
            if line == "\n":
                continue
            split_line = line.split(" ")
            key = split_line[0]
            val = split_line[1]
            # print(key)
            # print(val)
            if key == "problem_graph_size:":
                full_problem_graph_size.append(int(val))
            if key == "graph_nodes:":
                full_nodes.append(int(val))
            if key == "max_treewidth:":
                full_max_treewidth.append(int(val))
            if key == "Total_time:":
                full_total_time.append(float(val))
            if key == "Num_bucket_contractions:":
                full_num_bucket_contractions.append(float(val))

    # plt.plot(problem_graph_size, diag_nodes, 'r--', problem_graph_size, full_nodes, 'bs')

    plt.figure()

    plt.subplot(311)
    plt.xlabel('Problem graph size')
    plt.ylabel('tensor graph nodes')
    plt.plot(diag_problem_graph_size, diag_nodes, 'bo', full_problem_graph_size, full_nodes, 'rs')

    plt.subplot(312)
    plt.xlabel('Problem graph size')
    plt.ylabel('treewidth')
    plt.plot(diag_problem_graph_size, diag_max_treewidth, 'bo', full_problem_graph_size, full_max_treewidth, 'rs')

    plt.subplot(313)
    plt.xlabel('Problem graph size')
    plt.ylabel('runtime')
    plt.plot(diag_problem_graph_size, diag_total_time, 'bo', full_problem_graph_size, full_total_time, 'rs')

    # plt.plot(diag_problem_graph_size, diag_nodes, 'bo', full_problem_graph_size, full_nodes, 'rs')

    plt.show()

# performance_backend_simulation()
plot_simulation_results()
