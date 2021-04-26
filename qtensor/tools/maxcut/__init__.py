import numpy as np
import networkx as nx

from .goemans_williamson import gw_cost
from .gurobi import solve_maxcut as gurobi_maxcut

def check_sol(G: nx.Graph, sol: list):
    """
    Solution values should be 0 or 1
    """
    assert len(sol) == G.number_of_nodes()
    valmap = {n: v for n, v in zip(G.nodes, sol)}
    cost = 0
    for i, j in G.edges:
        a, b = valmap[i], valmap[j]
        cost += a + b - 2*a*b
    return cost

def spectral_bound(G):
    """ Returns a spectral upper bound to MaxCut on graph G, fraction of edges. """
    def is_regular(G):
        degrees = np.array(G.degree)[:, 1]
        return not bool(np.sum(degrees - degrees.min()))
    if not is_regular(G):
        raise ValueError('Graph should be regular')
    A = nx.adjacency_matrix(G)
    d = G.degree(list(G.nodes())[0])
    M = 1/d*A.toarray()
    spectrum = np.linalg.eigvals(M)
    spectrum.sort()
    return 1/2 + 1/2*np.abs(spectrum[0])


def maxcut_optimal(G, print_stats=False):
    '''
    Computes the MAXCUT partition for a graph G

    G - networkX graph

    Args:
        G (networkx.Graph): Graph
    Returns:
        number of edges cut
        the partition (as signs +/-1)

    '''
    import time
    # There is a Z2 symmetry which takes Z -> -Z. We can break this symmetry and
    #  gain a factor of 2 speedup by fixing the last bit to be in the +1 sector
    Nbits = len(G.nodes) - 1

    '''
    Generate an indexing of the Hilbert space in the Z basis.
    '''
    # Index 0 is the all up state
    # Index -1 is the all down state
    # The least significant bit is the lefthand index in the bitarray (eg qubit 0)
    def index2state(index):
        return np.array([1 - 2*((index>>i)%2) for i in range(Nbits)])
    def state2index(state):
        return np.dot((1 - state)//2,2**np.arange(Nbits))

    t1 = time.time()

    indexing = np.arange(2**Nbits) # Making this not memory bound would be to iterate over chunks
    bits = [(1 - 2*((indexing>>i)%2)).astype(np.int8) for i in range(Nbits)]

    t2 = time.time()
    '''
    Generate the cost function by summing over all clauses
    '''
    # Note that because we have broken Z2 symmetry by fixing the sign of the last
    #  bit there needs to be a conditional on the summation (eg it is always 1)
    #  ZZ is a product of two Z expectation values.
    # In this way, any size clauses can be found (eg ZZZ is bits[]*bits[]*bits[])
    ZZcost = np.zeros(2**Nbits)
    for edge in G.edges:
        edge = np.sort(edge)
        if edge[1]==Nbits:
            ZZcost += (1 - bits[edge[0]])//2
        else:
            ZZcost += (1 - bits[edge[0]]*bits[edge[1]])//2

    t3 = time.time()
    # MAXCUT normally wants the max cut (huh.)
    maxdex = np.argmax(ZZcost)
    maxstate = np.concatenate((index2state(maxdex),[1]))

    t4 = time.time()

    if print_stats:
        print('Time to generate basis: {:0.4f}'.format(t2-t1))
        print('Time to generate cost:  {:0.4f}'.format(t3-t2))
        print('Time to find maximum:   {:0.4f}'.format(t4-t3))
        print('Total Time:             {:0.4f}'.format(t4-t1))

    return ZZcost[maxdex],maxstate

