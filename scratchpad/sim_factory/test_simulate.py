from Simulate import QtreeSimulate

import sys
sys.path.append('..')
import utils_qaoa
import utils


def test_qtree():
    qc, N = utils_qaoa.get_test_qaoa(20, 2, type='randomreg', degree=3)
    sim = QtreeSimulate(qc)
    assert N == sim.n_qubits
    result = sim.simulate()
    print(result.data)
    assert result
