import qensor
from qensor import CirqQAOAComposer, QtreeQAOAComposer
from qensor import QAOAQtreeSimulator
from qensor.Simulate import CirqSimulator, QtreeSimulator
from qensor.FeynmanSimulator import FeynmanSimulator
import numpy as np
import networkx as nx


def get_test_problem(n=14, p=2, d=3):
    w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
    G = nx.from_numpy_matrix(w)

    G = nx.random_regular_graph(d, n)
    gamma, beta = [np.pi/3]*p, [np.pi/2]*p
    return G, gamma, beta

def test_qtree():
    G, gamma, beta = get_test_problem()

    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    print(composer.circuit)
    sim = QtreeSimulator()
    result = sim.simulate(composer.circuit)
    print(result)
    qtree_amp = result

    composer = CirqQAOAComposer(
        graph=G, gamma=[np.pi/3], beta=[np.pi/4])
    composer.ansatz_state()

    print(composer.circuit)
    sim = CirqSimulator()
    result = sim.simulate(composer.circuit)
    print(result)
    final_cirq = result.final_state
    assert final_cirq[0] - qtree_amp < 1e-5

    assert result


def test_parallel_batched():
    G, gamma, beta = get_test_problem(42, 3, 3)
    batch_vars = 2

    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    sim = FeynmanSimulator()
    result = sim.simulate(composer.circuit, batch_vars=batch_vars)
    print(result)

    total_amps = 2**G.number_of_nodes()
    batch_amps = 2**batch_vars
    assert len(result) == batch_amps
    sum_amps = np.sum(np.abs(np.square(result)))
    print('Sum amps', sum_amps)
    print('Amp ratio', batch_amps/total_amps )
    print('Total amps cnt', total_amps )
    #assert abs(sum_amps - batch_amps/total_amps ) < 1e-6


def test_qtree_energy():
    G, gamma, beta = get_test_problem()

    sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    E = sim.energy_expectation(
        G=G, gamma=gamma, beta=beta)

    print('Energy', E)
    assert np.imag(E)<1e-6

    E = np.real(E)

    Ed = G.number_of_edges()
    C = (Ed - E)/2

    print("Edges", Ed)
    print("Cost", C)
    assert E

if __name__ == "__main__":
    #test_qtree_energy()
    test_parallel_batched()
