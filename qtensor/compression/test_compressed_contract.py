from qtensor.compression import compressed_contract, compressed_sum, CompressedTensor, Tensor
from qtensor.compression import NumpyCompressor
from qtree.optimizer import Var
import numpy as np


def test_compressed_contract():
    A_ixs = [Var(x) for x in [8,7,6,5,4,3, 2]]
    A_comp = [Var(x) for x in [8, 7, 6]]
    B_ixs = [Var(x) for x in [10, 9, 3, 4, 2]]
    contract_ixs = [Var(x) for x in [3,2]]

    A_data = np.ones(2**len(A_ixs))
    #A_data = np.random.randn(2**len(A_ixs))
    A_data = A_data.reshape(*(v.size for v in A_ixs))
    A_data[1, 1] *= 2
    A_data[0, 1] *= 2
    A_data[:, :, :, 1] *= 1.2
    B_data = np.ones(2**len(B_ixs))*1.2
    #B_data = np.random.randn(2**len(B_ixs))*1.2
    B_data = B_data.reshape(*(v.size for v in B_ixs))

    A = CompressedTensor('A', A_ixs, data=A_data)
    A.compress_indices(A_comp)
    B = Tensor('B', B_ixs, data=B_data)
    print(f"Tensor A: {A}")
    print(f"Tensor B: {B}")

    res_ixs = list(set(A_ixs).union(B_ixs) - set(contract_ixs))
    res_ixs.sort(key=int, reverse=True)
    res = compressed_contract(A, B, res_ixs, contract_ixs,
                              mem_limit=3)
    print(f"Resulting Tensor: {res}")

    res = compressed_contract(A, B, res_ixs, contract_ixs,
                              mem_limit=10)

    print(f"Resulting Tensor: {res}")
    print(res.get_chunk(()).flatten())


    A_str = ''.join(chr(97+int(v)) for v in A_ixs)
    B_str = ''.join(chr(97+int(v)) for v in B_ixs)
    C_str = ''.join(chr(97+int(v)) for v in res_ixs)
    expr = f"{A_str},{B_str}->{C_str}"
    C = np.einsum(expr, A_data, B_data)
    print(f"Ground truth:")
    print( C.flatten())
    
    assert np.allclose(C, res.get_chunk(()))
    print("Success!")

def test_compressed_sum():
    A_ixs = [Var(x) for x in [8,7,6,5,4,3, 2]]
    A_comp = [Var(x) for x in [8, 7, 6]]
    A_data = np.random.rand(2**len(A_ixs))
    #A_data = np.random.randn(2**len(A_ixs))
    A_data = A_data.reshape(*(v.size for v in A_ixs))
    A = CompressedTensor('A', A_ixs, data=A_data)
    A.compress_indices(A_comp)
    sum_indices = [Var(i) for i in [2, 4]]

    res = compressed_sum(A, sum_indices, NumpyCompressor(), mem_limit=4)
    print(f"Resulting Tensor: {res}")
    res_ref = np.sum(A_data, axis=tuple(A_ixs.index(i) for i in sum_indices))
    assert np.allclose(res.get_chunk((0, )), res_ref[0])
    assert not np.allclose(res.get_chunk((1, )), res_ref[0])

    res = compressed_sum(res, [Var(5)], NumpyCompressor(), mem_limit=4)
    assert isinstance(res, Tensor)
    assert np.allclose(res.data, res_ref.sum(axis=3))


if __name__=="__main__":
    test_compressed_contract()
