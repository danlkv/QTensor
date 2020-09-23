from qtensor import CirqQAOAComposer, QtreeQAOAComposer
from qtensor import QiskitQAOAComposer
from qtensor import QtreeSimulator
from qtree.operators import from_qiskit_circuit

import networkx as nx
import numpy as np

import cirq

def get_test_problem():
    w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
    G = nx.from_numpy_matrix(w)
    gamma, beta = [np.pi/3], [np.pi/2]
    return G, gamma, beta


def test_cirq_sim():
    G, gamma, beta = get_test_problem()

    composer = CirqQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    composer.ansatz_state()
    sim = cirq.Simulator()
    result = sim.simulate(composer.circuit)

    print(result)
    assert result
    assert composer.n_qubits == G.number_of_nodes()


def test_qtree_smoke():
    G, gamma, beta = get_test_problem()

    composer = QtreeQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    composer.ansatz_state()

    print(composer.circuit)
    assert composer.circuit
    assert composer.n_qubits == G.number_of_nodes()

    composer = QtreeQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    composer.energy_expectation_lightcone(list(G.edges())[0])

    print(composer.circuit)
    assert composer.circuit
    assert composer.n_qubits == G.number_of_nodes()

def test_cirq_smoke():
    G, gamma, beta = get_test_problem()

    composer = CirqQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    composer.ansatz_state()

    print(composer.circuit)
    assert composer.circuit
    assert composer.n_qubits == G.number_of_nodes()

def test_qiskit_smoke():
    G, gamma, beta = get_test_problem()

    composer = QiskitQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    composer.ansatz_state()

    print(composer.circuit)
    assert composer.circuit
    assert composer.n_qubits == G.number_of_nodes()

def test_qiskit_convert():
    G, gamma, beta = get_test_problem()

    qiskit_com = QiskitQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    qiskit_com.ansatz_state()

    # Convert Qiskit circuit to Qtree circuit
    n, qc = from_qiskit_circuit(qiskit_com.circuit)
    sim = QtreeSimulator()
    all_gates = sum(qc, [])
    # Simulate converted circuit
    first_amp_from_qiskit = sim.simulate(all_gates)

    com = QtreeQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    com.ansatz_state()
    # Simulate same circuit but created by Qtree composer
    first_amp_orig = sim.simulate(com.circuit)
    assert np.allclose(*[np.abs(x) for x in (first_amp_from_qiskit, first_amp_orig)])
    assert np.allclose(first_amp_from_qiskit, first_amp_orig)

if __name__ =='__main__':
    test_qtree_smoke()
