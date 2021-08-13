import networkx as nx
import pytest
import numpy as np
from qtensor import QAOA_energy, QAOAQtreeSimulatorSymmetryAccelerated, QtreeQAOAComposer

@pytest.mark.skip(reason='pynauty throws "Illegal instruction" on gh actions')
def test_parallel_lightcone():
    from cartesian_explorer import parallels
    from qtensor.tools import lightcone_orbits
    par = parallels.Multiprocess(processes=2)
    G = nx.random_regular_graph(3, 1000, seed=42)
    eorbits_dict = lightcone_orbits.get_edge_orbits_lightcones(G, p=3, nprocs=par)

#@pytest.mark.skip(reason='pynauty throws "Illegal instruction" on gh actions')
def test_lightcone_energy_value():
    G = nx.random_regular_graph(3, 100, seed=42)
    gamma, beta = [np.pi/3], [np.pi/2]

    E = QAOA_energy(G, gamma, beta)

    sym = QAOAQtreeSimulatorSymmetryAccelerated(QtreeQAOAComposer)

    E2 = sym.energy_expectation(G, gamma, beta)

    assert(np.isclose(E,E2))

#@pytest.mark.skip(reason='pynauty throws "Illegal instruction" on gh actions')
def test_lightcone_energy_value_large():
    # large graph should trigger parallel evaluation of lightcone orbits
    G = nx.random_regular_graph(3, 1000, seed=42)
    gamma, beta = [np.pi/3], [np.pi/2]

    E = QAOA_energy(G, gamma, beta)

    sym = QAOAQtreeSimulatorSymmetryAccelerated(QtreeQAOAComposer)

    E2 = sym.energy_expectation(G, gamma, beta, nprocs=8)

    assert(np.isclose(E,E2))
