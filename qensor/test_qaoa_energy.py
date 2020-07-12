from qensor import QAOA_energy
from qensor.Simulate import CirqSimulator, QtreeSimulator
import numpy as np
import networkx as nx

def get_test_problem():
    w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
    G = nx.from_numpy_matrix(w)

    G = nx.random_regular_graph(5, 14)
    gamma, beta = [np.pi/3], [np.pi/2]
    return G, gamma, beta

def test_qaoa_energy():
    G, gamma, beta = get_test_problem()
    res = QAOA_energy(G, gamma, beta)
    print('result', res)
    assert res
