import numpy as np
import networkx as nx
import qtensor as qt
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor.Simulate import QtreeSimulator
from qtensor.FeynmanSimulator import FeynmanSimulator
from qtensor import QtreeQAOAComposer

G = nx.random_regular_graph(3, 10)
gamma, beta = [np.pi/3], [np.pi/2]

composer = QtreeQAOAComposer(graph=G, gamma=gamma, beta=beta) # I would expect composer.circuit to have a meaningful default
composer.ansatz_state()

tn = QtreeTensorNet.from_qtree_gates(composer.circuit)

# Can slice the tn with an empty dictionary like this
# Should that error? 
sliced = tn.slice({})

# Is there a valid use case of getting the slice dict from a simulator like this, for which _get_slice_dict should be public?
# Why is prepare_buckets public when _get_slice_dict isn't? I don't see the use case for calling prepare_buckets externally
from qtensor.Simulate import QtreeSimulator

# I feel like Simulator initializer should take an argument. If the primary method on a simulator, and that accepts a circuit, why not just init it with a circuit?
sim = FeynmanSimulator()
# Beside this exercise, is there a reason to get slice_dicts from a simulator?
# I don't see an easy way to get slice_dict from a simulator, and the simulator seems the only way to get it
# Also, how do you test changes. Do you always push before rebuilding the container?
result = sim.simulate(composer.circuit)

# Not clear on how to contract the tn?
# Qtree Basic Usage notebook shows how to contract using npfr. Is that what is meant? Or is there a QTensor api I'm missing?