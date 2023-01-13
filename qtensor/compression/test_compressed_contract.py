from qtensor.compression import compressed_contract, CompressedTensor, Tensor
from qtree.optimizer import Var
import numpy as np


def test_compressed_contract():
    A_ixs = [Var(x) for x in [8,7,6,5,4,3, 2]]
    A_comp = [Var(x) for x in [8, 7, 6]]
    B_ixs = [Var(x) for x in [9, 3, 4, 2]]
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
    A.slice(A_comp)
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


if __name__=="__main__":
    test_compressed_contract()
