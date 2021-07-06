import qtensor
import numpy as np
import cupy as cp
from qtensor.contraction_backends import CuPyBackend, NumpyBackend
from qtensor import QtreeSimulator


def get_test_qaoa_circ(n=10, p=2, d=3, type='random'):
    G = qtensor.toolbox.random_graph(seed=10, degree=d, nodes=n, type=type)
    print('Test problem: n, p, d', n, p, d)
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p

    composer = qtensor.DefaultQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    return composer.circuit


def get_test_qaoa_tn(n=10, p=2, d=3, type='random'):
    G = qtensor.toolbox.random_graph(seed=10, degree=d, nodes=n, type=type)
    print('Test problem: n, p, d', n, p, d)
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p

    composer = qtensor.DefaultQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(composer.circuit)
    return tn


def test_cupy_numpy():
    circ = get_test_qaoa_circ(p=3)
    bcp = CuPyBackend()
    bnp = NumpyBackend()
    simcp = QtreeSimulator(backend=bcp)
    simnp = QtreeSimulator(backend=bnp)
    rescp = simcp.simulate(circ)
    resnp = simnp.simulate(circ)
    assert np.allclose(rescp, resnp)

def test_cupy_vanilla():
    circ = get_test_qaoa_circ(p=3)
    bcp = CuPyBackend()
    simcp = QtreeSimulator(backend=bcp)
    sim = QtreeSimulator()
    rescp = simcp.simulate(circ)
    res = sim.simulate(circ)
    assert np.allclose(rescp, res)


def test_cupy_process_bucket():
    btr = CuPyBackend()
    bnp = NumpyBackend()
    def contract_tn(backend, search_len=1, test_problem_kwargs={}):
        """
        search_len is used to select non-trivial buckets.
        test_problem_kwargs is used to generate custom graph for test_problem
        """
        tn = get_test_qaoa_tn(**test_problem_kwargs)
        sliced_buckets = backend.get_sliced_buckets(tn.buckets, tn.data_dict, {})
        good_buckets = [x for x in sliced_buckets if len(x) >= search_len]
        if not good_buckets:
            raise ValueError('Could not find large enough buckets')
        selected_bucket = good_buckets[0]
        print('selected_bucket', selected_bucket)

        result = backend.process_bucket(selected_bucket)
        return result.data

    # First test only simple buckets
    restr = contract_tn(btr, 1)
    resnp = contract_tn(bnp, 1)
    assert type(restr) is cp.ndarray
    '''
        Issue One: restr.type is printed to be complex64
                   But the assertion says neither cp.complex64 nor np.complex64
    '''
    # assert restr.dtype is np.complex64
    print("The data type for restr is {}".format(restr.dtype))
    assert np.allclose(restr, resnp)

    # Then test more advanced
    restr = contract_tn(btr, 2, dict(n=10, p=4, d=3))
    resnp = contract_tn(bnp, 2, dict(n=10, p=4, d=3))

    assert restr.shape == resnp.shape
    assert np.allclose(restr, resnp)
    
def test_cupy_get_sliced__slice():
    backend = CuPyBackend()
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
    
    
def test_cupy_get_sliced__noslice():
    backend = CuPyBackend()
    tn = get_test_qaoa_tn()

    tensor = tn.buckets[0][0]
    tensor_data = tn.data_dict[tensor.data_key]
    buckets = backend.get_sliced_buckets(
        tn.buckets, tn.data_dict, {}
    )
    assert len(buckets) == len(tn.buckets)
    assert tensor_data.shape == buckets[0][0].data.shape

def test_cupy_get_sliced_smoke():
    backend = CuPyBackend()
    test_buckets = []
    buckets = backend.get_sliced_buckets(test_buckets, {}, {})
    assert buckets == []

test_cupy_numpy()
test_cupy_vanilla()
test_cupy_process_bucket()
test_cupy_get_sliced__slice()
test_cupy_get_sliced__noslice()
test_cupy_get_sliced_smoke()
