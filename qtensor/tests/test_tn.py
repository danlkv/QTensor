import qtensor
import pytest
import numpy as np
import qtree

class InitSim(qtensor.QtreeSimulator):
    def set_init_state(self, state):
        self.init_state = state

    def _create_buckets(self):
        self.tn = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(
            self.all_gates, init_state=self.init_state, backend=self.backend,
        )
        self.tn.backend = self.backend

    def _get_slice_dict(self, initial_state=0, target_state=0):
        slice_dict = {}
        slice_dict.update(qtree.utils.slice_from_bits(target_state, self.tn.bra_vars))
        slice_dict.update(qtree.utils.slice_from_bits(target_state, self.tn.bra_vars))
        slice_dict.update({var: slice(None) for var in self.tn.free_vars})
        slice_dict.update({var: slice(None) for var in self.tn.ket_vars})
        return slice_dict

def test_init_state_1qbit():
    ops = qtensor.OpFactory.QtreeBuilder.operators
    TN = qtensor.optimisation.QtreeTensorNet

    qc = [ops.H(0)]
    _a = .5*np.sqrt(2)
    init_state = np.array([1, .5])
    ##
    #
    # The resulting array will be a matrix multiplication
    #
    #   ⎛ 1.5a ⎞ _ ⎛ a   a ⎞ ⎛ 1   ⎞
    #   ⎝ 0.5a ⎠ ‾ ⎝ a  -a ⎠ ⎝ 0.5 ⎠
    #
    #   where a = 0.5√2.
    #
    ##

    tn = TN.from_qtree_gates(qc, init_state=init_state)
    print(tn.buckets)

    sim = InitSim()
    sim.set_init_state(init_state)
    res = sim.simulate_batch(qc, batch_vars=1)
    print(res)

    assert np.allclose(np.array([1.5, .5])*_a, res)

def test_init_state():
    ops = qtensor.OpFactory.QtreeBuilder.operators
    TN = qtensor.optimisation.QtreeTensorNet
    N = 5
    qc = []
    for i in range(N):
        qc.append(ops.H(i))


    init_state = np.random.rand(2**N)

    tn = TN.from_qtree_gates(qc, init_state=init_state)
    print(tn.buckets)

    gate_mx = [op.gen_tensor() for op in qc]
    matrix = gate_mx[0]
    for mx in gate_mx[1:]:
        matrix = np.kron(matrix, mx)
    print('matrix H^{⊗N}', matrix)

    sim = InitSim()
    sim.set_init_state(init_state)
    res = sim.simulate_batch(qc, batch_vars=N)
    print('result\n', res)
    ref = matrix.dot(init_state)
    print('matrix⋅init_state\n', ref)

    assert np.allclose(res, ref)
