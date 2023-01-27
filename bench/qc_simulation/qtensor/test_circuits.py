import qtensor
import numpy as np
import networkx as nx

def get_qaoa_graph_params(n=10, p=2, d=3, type='random', seed=10):
    if type == 'random':
        G = nx.random_regular_graph(d, n, seed=seed)
    elif type == 'grid2d':
        G = nx.grid_2d_graph(n,n)
    elif type == 'line':
        G = nx.Graph()
        G.add_edges_from(zip(range(n-1), range(1, n)))
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p
    return G, gamma, beta

def gen_qaoa_maxcut_circuit(n=10, p=2, d=3, type='random', seed=10):
    G, gamma, beta = get_qaoa_graph_params(n, p, d, type, seed)
    composer = qtensor.QtreeQAOAComposer(graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    return composer.circuit
