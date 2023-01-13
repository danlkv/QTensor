import numpy as np

from qtensor.compression import CompressedTensor
from .CompressedTensor import Tensor, iterate_indices

def compressed_contract(A:Tensor, B: Tensor,
                        result_ixs, contract_ixs,
                        mem_limit):
    all_indices = list(set(A.indices).union(B.indices))
    all_indices.sort(key=int, reverse=True)
    result_indices = list(set(all_indices) - set(contract_ixs))
    result_indices.sort(key=int, reverse=True)
    to_small_int = lambda x: all_indices.index(x)

    exist_compressed = []
    for T in [A, B]:
        if isinstance(T, CompressedTensor):
            exist_compressed += T.slice_indices

    exist_compressed.sort(key=int, reverse=True)
    need_compressed = result_indices[:-mem_limit]
    print(f"Need compression: {need_compressed}")
    add_compress = set(need_compressed) - set(exist_compressed)
    remove_compress = set(exist_compressed) - set(need_compressed)
    retain_compress = set(exist_compressed).intersection(need_compressed)
    add_compress = list(add_compress)
    remove_compress = list(remove_compress)
    retain_compress = list(retain_compress)

    remove_compress.sort(key=int, reverse=True)
    retain_compress.sort(key=int, reverse=True)
    add_compress.sort(key=int, reverse=True)

    newT_name= 'C'+str(int(all_indices[0]))

    if len(need_compressed)==0 and len(exist_compressed)==0:
        A_ixs = list(map(int, A.indices))
        B_ixs = list(map(int, B.indices))
        
        result = np.einsum(A.data, A_ixs, B.data, B_ixs)
        return Tensor(newT_name, all_indices, data=result)

    R = CompressedTensor(newT_name,
                         result_indices,
                         slice_indices=need_compressed
                        )

    chunk_ixs = list(set(result_indices) - set(need_compressed))
    chunk_ixs.sort(key=int, reverse=True)
    print(f"Chunk indices: {chunk_ixs}")
    print(f"Add compression: {add_compress}, Retain compression: {retain_compress}, remove_compress: {remove_compress}")
    slice_dict = {i: slice(None) for i in all_indices}
    for iadd in iterate_indices(add_compress):
        for iret in iterate_indices(retain_compress):

            chunk = np.empty(2**len(chunk_ixs), dtype=B.data.dtype)
            chunk = chunk.reshape(*(v.size for v in chunk_ixs))
            for irm in iterate_indices(remove_compress):
                for i, ival in zip(remove_compress, irm):
                    slice_dict[i] = ival#slice(ival, ival+1)
                chunk_view = chunk[tuple(
                    slice_dict[i] for i in chunk_ixs
                )]
                if isinstance(A, CompressedTensor):
                    A_data = A.get_chunk(iret+irm)
                    A_ixs = A.array_indices
                else:
                    A_data = A.data
                    A_ixs = A.indices

                # TODO: handle when A and B are sliced differently
                if isinstance(B, CompressedTensor):
                    B_data = B.get_chunk(iret+irm)
                    B_ixs = B.array_indices
                else:
                    B_data = B.data
                    B_ixs = B.indices
                # --
                for ia, iaval in zip(add_compress, iadd):
                    slice_dict[ia] = iaval#slice(iaval, iaval+1)
                ixsa = set(add_compress).intersection(B_ixs)
                if len(ixsa):
                    B_data = B_data[tuple(
                        slice_dict[i] for i in B_ixs
                    )]
                    for _del in ixsa:
                        B_ixs = tuple(i for i in B_ixs if i!=_del)

                A_ixs = list(map(int, A_ixs))
                B_ixs = list(map(int, B_ixs))
                
                C_ixs = list(map(int, [v for v in chunk_ixs if v not in exist_compressed]))
                #print(f"A indices: {A_ixs}, B indices: {B_ixs}, C indices:{C_ixs}")
                A_str = ''.join(chr(97+int(v)) for v in A_ixs)
                B_str = ''.join(chr(97+int(v)) for v in B_ixs)
                C_str = ''.join(chr(97+int(v)) for v in C_ixs)
                expr = f"{A_str},{B_str}->{C_str}"
                #np.einsum(A_data, A_ixs, B_data, B_ixs, C_ixs, out=chunk_view)
                print(f"Expr: {expr}")
                np.einsum(expr, A_data, B_data, out=chunk_view)
            R.set_chunk(iadd+iret, chunk)
    return R



    


