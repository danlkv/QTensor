from qtensor import CirqQAOAComposer, QtreeQAOAComposer, DefaultQAOAComposer
from qtensor import QiskitQAOAComposer, ZZQtreeQAOAComposer
from qtensor import QtreeSimulator
import qtensor
from qtree.operators import from_qiskit_circuit
from functools import lru_cache

import networkx as nx
import numpy as np

import cirq
from qtensor.tests import get_test_problem


def test_cirq_sim():
    G, gamma, beta = get_test_problem()

    composer = CirqQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    sim = cirq.Simulator()
    result = sim.simulate(composer.circuit)

    print(result)
    assert result
    assert composer.n_qubits == G.number_of_nodes()


def test_non_chordal_lightcones():
    G, gamma, beta = get_test_problem()

    for edge in G.edges():
        composer1 = DefaultQAOAComposer(
            graph=G, gamma=gamma, beta=beta)

        composer2 = ZZQtreeQAOAComposer(
            graph=G, gamma=gamma, beta=beta)
        composer1.energy_expectation_lightcone(edge)
        composer2.energy_expectation_lightcone(edge)
        assert len(composer1.circuit) == len(composer2.circuit)

def test_qtree_default_smoke():
    G, gamma, beta = get_test_problem()

    composer = DefaultQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    print(composer.circuit)
    assert composer.circuit
    assert composer.n_qubits == G.number_of_nodes()

    composer = DefaultQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    edge = list(G.edges())[0]
    composer.energy_expectation_lightcone(edge)
    subg = qtensor.utils.get_edge_subgraph(G, edge, dist=len(gamma))

    print(composer.circuit)
    assert composer.circuit
    assert composer.n_qubits == subg.number_of_nodes()

def test_qtree_smoke():
    G, gamma, beta = get_test_problem()

    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    print(composer.circuit)
    assert composer.circuit
    assert composer.n_qubits == G.number_of_nodes()

    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    edge = list(G.edges())[0]
    composer.energy_expectation_lightcone(edge)
    subg = qtensor.utils.get_edge_subgraph(G, edge, dist=len(gamma))

    print(composer.circuit)
    assert composer.circuit
    assert composer.n_qubits == subg.number_of_nodes()

def test_cirq_smoke():
    G, gamma, beta = get_test_problem()

    composer = CirqQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    print(composer.circuit)
    assert composer.circuit
    assert composer.n_qubits == G.number_of_nodes()

def test_qiskit_smoke():
    G, gamma, beta = get_test_problem()

    composer = QiskitQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    print(composer.circuit)
    assert composer.circuit
    assert composer.n_qubits == G.number_of_nodes()

def test_qiskit_convert():
    G, gamma, beta = get_test_problem()

    qiskit_com = QiskitQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    qiskit_com.ansatz_state()

    # Convert Qiskit circuit to Qtree circuit
    n, qc = from_qiskit_circuit(qiskit_com.circuit)
    sim = QtreeSimulator()
    all_gates = sum(qc, [])
    # Simulate converted circuit
    first_amp_from_qiskit = sim.simulate(all_gates)

    com = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    com.ansatz_state()
    # Simulate same circuit but created by Qtree composer
    first_amp_orig = sim.simulate(com.circuit)
    assert np.allclose(*[np.abs(x) for x in (first_amp_from_qiskit, first_amp_orig)])
    assert np.allclose(first_amp_from_qiskit, first_amp_orig)

def test_view():
    G, gamma, beta = get_test_problem()
    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    print(composer.circuit)
    assert composer.circuit
    other_builder = composer.builder.view()
    assert other_builder.circuit == composer.circuit

def test_copy():
    G, gamma, beta = get_test_problem()
    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    print(composer.circuit)
    assert composer.circuit
    other_builder = composer.builder.copy()
    assert other_builder.circuit == composer.circuit


def test_expectation():
    G, gamma, beta = get_test_problem()
    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    circ = composer.expectation(composer.builder.operators.H, 0)
    assert circ
    assert len(circ) == len(composer.circuit)*2 + 1

if __name__ =='__main__':
    test_qtree_smoke()
