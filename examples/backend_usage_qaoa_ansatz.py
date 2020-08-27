from qensor.optimisation.Optimizer import OrderingOptimizer
from qensor.optimisation.TensorNet import QtreeTensorNet
from qensor import QtreeQAOAComposer
from qensor import QAOAQtreeSimulator
from qensor import PerfNumpyBackend
from qensor import QAOA_energy
import networkx as nx
import numpy as np

G = nx.random_regular_graph(4, 20)
gamma, beta = [np.pi/3], [np.pi/2]

composer = QtreeQAOAComposer(
    graph=G, gamma=gamma, beta=beta)
composer.ansatz_state()

backend = PerfNumpyBackend(print=True)
sim = QAOAQtreeSimulator(composer, bucket_backend = backend)
sim.simulate(composer.circuit)
print(sim.bucket_backend)
print(backend._profile_results)
print(backend.gen_report())
