from Simulate import QtreeSimulate
from Simulate import TensoNetSimulate

import sys
sys.path.append('..')
import utils_qaoa
import utils

import pytest

@pytest.mark.skip # Just focus on tn for now
def test_qtree():
    qc, N = utils_qaoa.get_test_qaoa(20, 2, type='randomreg', degree=3)
    sim = QtreeSimulate(qc)
    assert N == sim.n_qubits
    result = sim.simulate()
    print(result.data)
    assert result

def test_tn():
    qc, N = utils_qaoa.get_test_qaoa(20, 2, type='randomreg', degree=3)
    sim = TensoNetSimulate(qc)
    assert N == sim.n_qubits
    result = sim.simulate()
    print(result.data)
    assert result
