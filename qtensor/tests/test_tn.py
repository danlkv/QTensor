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

def get_test_problem(n=14, p=2, d=3):
    G = qtensor.toolbox.random_graph(n, type='random', seed=13, degree=3)
    gamma, beta = [np.pi/3]*p, [np.pi/2]*p
    return G, gamma, beta

def test_match_qaoa():
    N = 10
    G, gamma, beta = get_test_problem(n=N)

    composer = qtensor.QtreeQAOAComposer(graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    sim1 = qtensor.QtreeSimulator()
    result_reference = sim1.simulate(composer.circuit)

    sim2 = InitSim()
    x = np.zeros(2**N)
    x[0] = 1
    sim2.set_init_state(x)
    result = sim2.simulate(composer.circuit)

    assert np.allclose(result, result_reference)

    x[-1] = 1
    x[0] = 0
    sim2.set_init_state(x)
    result = sim2.simulate(composer.circuit)

    assert not np.allclose(result, result_reference)

    sim1.set_init_state(2)
    result_reference = sim1.simulate(composer.circuit)

    sim2 = InitSim()
    x = np.zeros(2**N)
    x[2] = 1
    sim2.set_init_state(x)
    result = sim2.simulate(composer.circuit)

    assert np.allclose(result, result_reference)

class TorchComposer(qtensor.CircuitComposer):
    def _get_builder(self):
        return qtensor.TorchBuilder(n_qubits=self.params['n_qubits'])

class QtreeCaomposer(qtensor.CircuitComposer):
    def _get_builder(self):
        return qtensor.QtreeBuilder(n_qubits=self.params['n_qubits'])

def test_substitute_parameters():
    import torch
    import qtensor.optimisation.TensorNet as TN
    N = 2
    composer = QtreeCaomposer(n_qubits=N)
    # Numpy backend currently doesn't work, since the changes are not in-place
    #backend = qtensor.contraction_backends.NumpyBackend()
    #parameters = np.random.rand(N)
    backend = qtensor.contraction_backends.TorchBackend()
    parameters = torch.rand(N)
    def gen_circuit(composer, parameters):
        for i, x in enumerate(parameters):
            composer.apply_gate(composer.operators.XPhase, i, alpha=x)

    gen_circuit(composer, parameters)
    assert len(composer.circuit) == N
    tn = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(composer.circuit)
    print('tn.buckets', tn.buckets)
    opt = qtensor.toolbox.get_ordering_algo('greedy')
    peo, _ = opt.optimize(tn)
    print('peo', peo)
    print('len peo', len(peo), 'len tn', len(tn.buckets))
    TN.reorder_tn(tn, peo)
    print('tn_opt.buckets', tn.buckets)

    #--
    tn.backend = backend
    print('tdata', [t.data for b in tn.buckets for t in b])
    sliced = TN.slice_tn(tn)
    print('tdata', [t.data for b in sliced for t in b])
    res = qtree.optimizer.bucket_elimination(sliced, tn.backend.process_bucket)
    A = tn.backend.get_result_data(res).flatten()
    print('A', A)

    #--
    parameters[0] = .5
    print('taparams', [t.parameters for b in tn.buckets for t in b if hasattr(t,'parameters')])
    sliced = TN.repopulate_data(tn, composer.circuit)
    sliced = TN.slice_tn(tn)
    print('Btdata', [t.data for b in sliced for t in b])
    res = qtree.optimizer.bucket_elimination(sliced, tn.backend.process_bucket)
    B = tn.backend.get_result_data(res).flatten()
    print('B', B)
    assert not np.allclose(A, B)
    #assert False


