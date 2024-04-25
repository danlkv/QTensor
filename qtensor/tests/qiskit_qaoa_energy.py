import numpy as np
import networkx as nx
from functools import partial


def maxcut_obj(x, G):
    """
    Given a bitstring as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.

    Args:
        x: str
           solution bitstring
        G: networkx graph

    Returns:
        obj: float
    """
    obj = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            obj -= 1
    return obj


def compute_expectation(counts, G):
    """
    Computes expectation value based on measurement results

    Args:
        counts: dict
                key as bitstring, val as count

        G: networkx graph

    Returns:
        avg: float
             expectation value
    """
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        # Qiskit uses little-endian format, this is corrected from their official nb
        obj = maxcut_obj(bitstring[::-1], G)
        avg += obj * count
        sum_count += count
    return avg/sum_count


# We will also bring the different circuit components that
# build the qaoa circuit under a single function
def create_qaoa_circ(G, theta):
    """
    Creates a parametrized qaoa circuit

    Args:
        G: networkx graph
        theta: list
               unitary parameters

    Returns:
        qc: qiskit circuit
    """
    from qiskit import QuantumCircuit
    nqubits = len(G.nodes())
    p = len(theta)//2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)
    beta = theta[:p]
    gamma = theta[p:]
    # initial_state
    for i in range(0, nqubits):
        qc.h(i)
    for irep in range(0, p):
        # problem unitary
        for pair in list(G.edges()):
            qc.rzz(2 * gamma[irep], pair[0], pair[1])

        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)
    qc.measure_all()
    return qc

# Finally we write a function that executes the circuit on the chosen backend
def get_expectation(G, shots=512):
    """
    Runs parametrized circuit
    Args:
        G: networkx graph
        p: int,
           Number of repetitions of unitaries
    """
    from qiskit_aer import AerSimulator
    backend = AerSimulator(method='statevector')

    def execute_circ(theta):
        qc = create_qaoa_circ(G, theta)
        counts = backend.run(qc, seed_simulator=10,
                             shots=shots).result().get_counts()
        return compute_expectation(counts, G)
    return execute_circ

def simulate_qiskit_amps(G, gamma, beta, shots=100_000):
    assert len(gamma) == len(beta)
    p = len(gamma)
    theta = np.concatenate([np.array(beta), -np.array(gamma)/2])
    expectation = get_expectation(G, shots=shots)
    res = expectation(theta)
    return -res

def test_simulate_qiskit_amps():
    elist = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [5, 7], [5, 8], [6, 8], [6, 9], [7, 9]]
    G = nx.OrderedGraph()
    G.add_edges_from(elist)
    parameters = np.array([5.192253984583296, 5.144373231492732, 5.9438949617723775, 5.807748946652058, 3.533458907810596, 6.006206583282401, 6.122313961527631, 6.218468942101044, 6.227704753217614,

0.3895570099244132, -0.1809282325810937, 0.8844522327007089, 0.7916086532373585, 0.21294534589417236, 0.4328896243354414, 0.8327451563500539, 0.7694639329585451, 0.4727893829336214])
    beta = parameters[:9]
    gamma = -parameters[9:]

    result = simulate_qiskit_amps(G, gamma, beta)
    assert abs(abs(result) - 12) < 1e-2

if __name__ == "__main__":
    test_simulate_qiskit_amps()

