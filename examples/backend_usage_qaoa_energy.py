from qtensor.optimisation.Optimizer import OrderingOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor import QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor import PerfNumpyBackend
from qtensor import QAOA_energy
import networkx as nx
import numpy as np

G = nx.random_regular_graph(4, 20)
gamma, beta = [np.pi/3], [np.pi/2]

backend = PerfNumpyBackend(print=True)
sim = QAOAQtreeSimulator(QtreeQAOAComposer, bucket_backend = backend)
sim.energy_expectation(G, gamma, beta)

print(backend.gen_report())
