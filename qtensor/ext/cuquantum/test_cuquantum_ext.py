from qtensor.ext import cuquantum as cq
from qtensor.tests import get_test_problem
from qtensor import DefaultQAOAComposer
import random, numpy as np

random.seed(10)
np.random.seed(10)


import logging
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s %(levelname)-8s %(message)s', force=True )

def test_cuquantum():
    G, gamma, beta = get_test_problem(16, 3, d=3)
    composer = DefaultQAOAComposer(
            graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    circ = composer.circuit
    print('circ', circ)
    print('circ len', len(circ))
    tn = cq.CuTensorNet.from_qtree_gates(circ)
    slicer_opt = cq.cq.SlicerOptions()
    reconf_opt = cq.cq.ReconfigOptions(num_iterations=0)   
    print('Cuquantum equation', tn._eq)
    print('Cuquantum tensor count', len(tn))
    path, info = tn.net.contract_path(
        optimize=dict(
            samples=16,
            threads=1,
            slicing=slicer_opt,
            reconfiguration=reconf_opt
        )
    )