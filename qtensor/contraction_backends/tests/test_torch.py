import qtensor
import numpy as np
from qtensor.contraction_backends import TorchBackend

def get_test_qaoa_tn(n=10, p=2, d=3, type='random'):
    G = qtensor.toolbox.random_graph(seed=10, degree=d, nodes=n, type=type)
    print('Test problem: n, p, d', n, p, d)
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p

    composer = qtensor.DefaultQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(composer.circuit)
    return tn

def test_torch_get_sliced__slice():
    backend = TorchBackend()
    tn = get_test_qaoa_tn()
    tensor = tn.buckets[0][0]
    tensor_data = tn.data_dict[tensor.data_key]
    slice_dict = {
        tensor.indices[0]: slice(0, 1)
    }
    buckets = backend.get_sliced_buckets(
        tn.buckets, tn.data_dict, slice_dict
    )
    assert tensor_data.shape != buckets[0][0].data.shape

def test_torch_get_sliced__noslice():
    backend = TorchBackend()
    tn = get_test_qaoa_tn()

    tensor = tn.buckets[0][0]
    tensor_data = tn.data_dict[tensor.data_key]
    buckets = backend.get_sliced_buckets(
        tn.buckets, tn.data_dict, {}
    )
    assert len(buckets) == len(tn.buckets)
    assert tensor_data.shape == buckets[0][0].data.shape

def test_torch_get_sliced_smoke():
    backend = TorchBackend()
    test_buckets = []
    buckets = backend.get_sliced_buckets(test_buckets, {}, {})
    assert buckets == []

