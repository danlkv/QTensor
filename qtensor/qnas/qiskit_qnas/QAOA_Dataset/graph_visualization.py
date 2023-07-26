import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def matrices_to_graphs(matrix_list):
    g_list = []
    for matrix in matrix_list:
        array = np.array(matrix)
        g = nx.convert_matrix.from_numpy_matrix(array)
        g_list.append(g)

    return g_list


graphs_file = open("20_10_node_erdos_renyi_graphs.txt")
matrix_list = np.loadtxt(graphs_file).reshape(20, 10, 10)
graph_list = matrices_to_graphs(matrix_list)

for i in range(20):
    nx.draw(graph_list[i])
    plt.show()
