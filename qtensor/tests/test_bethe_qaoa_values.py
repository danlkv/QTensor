import qtensor
import numpy as np
import pytest
from qtensor.toolbox import bethe_graph
from qtensor.tools import BETHE_QAOA_VALUES

@pytest.mark.parametrize("p", [1, 2, 3])
def test_bethe_qaoa_values(p):
    gammabeta = np.array(BETHE_QAOA_VALUES[str(p)]['angles'])
    gamma, beta = gammabeta[:p], gammabeta[p:]
    AR = BETHE_QAOA_VALUES[str(p)]['val']
    graph = bethe_graph(p, 3)
    print(f"Number of nodes in Bethe graph: {graph.number_of_nodes()}")
    sim = qtensor.QAOAQtreeSimulator(qtensor.QtreeQAOAComposer)
    eZZ = sim._get_edge_energy(graph, gamma/np.pi, beta/np.pi, edge=(0, 1))
    e_nocut = (1+eZZ)/2
    AR_sim = 1-e_nocut
    assert abs(AR_sim - AR) < 1e-6, f"p={p}, res={AR_sim}, AR={AR}"

if __name__ == "__main__":
    test_bethe_qaoa_values(1)
    test_bethe_qaoa_values(2)
    test_bethe_qaoa_values(3)