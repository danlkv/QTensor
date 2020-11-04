import numpy as np
import pytest

from qtensor.OpFactory import ZZ
from qtensor import ZZQtreeQAOAComposer as QtreeQAOAComposer
from qtensor.simplify_circuit import simplify_qtree_circuit
from qtensor.simplify_circuit import get_simplifiable_circuit_composer

def test_simplify_circuit():
    import networkx as nx
    p = 1
    G = nx.random_regular_graph(3, 300)
    gamma, beta = [np.pi/3]*p, [np.pi/2]*p

    class CustomComposer(QtreeQAOAComposer):
        def energy_edge(self, i, j):
            u, v = self.qubits[i], self.qubits[j]
            self.apply_gate(self.operators.ZZ, u, v, alpha=187)

    comp = CustomComposer(G, gamma=gamma, beta=beta)
    comp.energy_expectation_lightcone(list(G.edges)[0])
    print('comp circuit', comp.circuit)
    #[ print(x) for x in comp.circuit]
    new_circ = simplify_qtree_circuit(comp.circuit)
    #[ print(x) for x in new_circ]
    assert len(new_circ) <= len(comp.circuit)

@pytest.mark.skip(reason="work in progress")
def test_simplifiable_composer():
    comp = get_simplifiable_circuit_composer(200, p, 3)
    comp.energy_expectation_lightcone(list(G.edges)[0])
    print('comp circuit gen', comp.circuit)

if __name__ == "__main__":
    test_simplify_circuit()
