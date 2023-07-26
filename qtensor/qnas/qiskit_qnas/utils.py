import numpy as np
from collections import OrderedDict
import networkx as nx

# def save_ckt_to_disk(qckt, file):
#     ''' Serialize circuit to file'''
#     t = qckt.qtape.to_openqasm()
#     with open(file, 'w') as ofile:
#         ofile.write(t)


# def load_ckt_from_disk(file):
#     '''Load serialized ckt from file'''
#     with open(file, 'r') as ifile:
#         tape = ifile.read()
    
#     ckt = qml.from_qasm(tape)
#     return ckt


def get_graphs(graphs_file, energies_file):
    # graphs_file = open("QAOA_Dataset/20_10_node_erdos_renyi_graphs.txt")
    matrix_list = np.loadtxt(graphs_file).reshape(20, 10, 10)
    graph_list = matrices_to_graphs(matrix_list)
    energies = [int(l.strip()) for l in open(energies_file, 'r').readlines()]
    return graph_list, energies


def matrices_to_graphs(matrix_list):
    g_list = []
    for matrix in matrix_list:
        array = np.array(matrix)
        g = nx.from_numpy_array(array)
        g_list.append(g)

    return g_list


def get_adj_mat(G):
    n = G.number_of_nodes()
    adj = np.zeros((n, n))
    for e in G.edges:
        if nx.is_weighted(G):
            adj[e[0]][e[1]] = G[e[0]][e[1]]['weight']
            adj[e[1]][e[0]] = G[e[0]][e[1]]['weight']
        else:
            adj[e[0]][e[1]] = 1
            adj[e[1]][e[0]] = 1
    
    return adj

def decimal_to_binary(n):
    """converts decimal to binary and removes the prefix (0b)"""
    return bin(n).replace("0b", "")

def create_counts_dict(num_qubits: int, big_endian: bool):
    """Hxelper function for attach_qubit_names(). We create the dictionary that will eventually hold the probabilties."""

    # get the binary length of num_qubits. length is used so that the length of each key is the same
    length = int(np.ceil(np.log(2**num_qubits + 1)/np.log(2)) - 1)
    counts = OrderedDict()
    for i in range(2**num_qubits):
        # convert to binary and then pad the right side with 0s
        if big_endian == True:
            key_i = str(decimal_to_binary(i).zfill(length))
            key_i[::-1]
            counts[key_i] = 0
        else:
            key_i = str(decimal_to_binary(i).zfill(length))
            counts[key_i] = 0
    return counts


def attach_qubit_names(probs_list: list, big_endian: bool = True):
    """Creates a dictionary of qubit names and probabilities from a probability list. 

    Args:
        probs_list (list): the list probabilities we are turning into a dictionary
        big_endian (bool): the order that qubits are displayed.  
                           use big_endian = False if you want the qubit ordering that Qiskit uses
    
    Returns: 
        probs_dict (dict): a dictionary of qubit name (int): probability (float)
    """

    num_qubits = int(np.log2(len(probs_list)))
    probs_dict = create_counts_dict(num_qubits, big_endian)
    for key, prob in zip(probs_dict, probs_list):
        probs_dict[key] = prob
    return probs_dict