import numpy as np

from qtensor.compression import CompressedTensor
from .CompressedTensor import Tensor, iterate_indices
from .CompressedTensor import Compressor

# taken from numpy/core/einsumfunc.py
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)

def contract_two_tensors(A, B, T_out):
    """
    Contract tensors A and B along their common indices and write result to T_out.
    T_out tensor should be pre-allocated with data.

    This takes care of the case where indices of A and B are Vars with large integer id
    """
    result_indices = T_out.indices
    out_buffer = T_out.data
    max_id = 0
    A_ints = []
    B_ints = []

    for a_i in A.indices:
        a_int = int(a_i)
        max_id = max(max_id, a_int)
        A_ints.append(a_int)

    for b_i in B.indices:
        b_int = int(b_i)
        max_id = max(max_id, b_int)
        B_ints.append(b_int)

    if max_id > len(einsum_symbols):
        # -- relabel indices to small ints
        all_indices = set(A_ints + B_ints)
        relabel_dict_int = {i: j for j, i in enumerate(all_indices)}
        A_ints = [relabel_dict_int[i] for i in A_ints]
        B_ints = [relabel_dict_int[i] for i in B_ints]
        result_ints = [relabel_dict_int[int(i)] for i in result_indices]
    else:
        result_ints = list(map(int, result_indices))

    np.einsum(A.data, A_ints, B.data, B_ints, result_ints, out=out_buffer)


def compressed_contract(A:Tensor, B: Tensor,
                        contract_ixs, mem_limit,
                        compressor:Compressor):
    """
    Contract tensors A and B along `contract_ixs` and return the result

    The result tensor indices will be ordered from largest to smallest
    """
    all_indices = list(set(A.indices).union(B.indices))
    all_indices.sort(key=int, reverse=True)
    result_indices = list(set(all_indices) - set(contract_ixs))
    result_indices.sort(key=int, reverse=True)
    to_small_int = lambda x: all_indices.index(x)

    # -- Find set of existing compressed that will be decompressed
    exist_compressed = set()
    for T in [A, B]:
        if isinstance(T, CompressedTensor):
            exist_compressed.update(T.slice_indices)
    # In this particular case, we need not to sort these indices,
    # since the iteration over fast index gives same latency as over slow index
    # Potential improvement: if A_S and B_S are different, run outer loop 
    # over min(A_S, B_S) and inner over the rest indices. This will reduce 
    # the number of decompressions.
    # --


    need_compressed = result_indices[:-mem_limit]
    print(f"Need compression: {need_compressed}")
    new_tensor_name = 'C'+str(int(all_indices[0]))

    # -- Early return: if no need to compress, do the regular contraction
    if len(need_compressed)==0 and len(exist_compressed)==0:
        C = Tensor.empty(new_tensor_name, result_indices)
        contract_two_tensors(A, B, C)
        return C
    # --

    remove_compress = exist_compressed - set(need_compressed)
    R = CompressedTensor(new_tensor_name,
                         result_indices,
                         slice_indices=need_compressed,
                         compressor=compressor
                        )

    result_chunk_ixs = result_indices[-mem_limit:]
    print(f"Chunk indices: {result_chunk_ixs}, remove_compress: {remove_compress}")
    slice_dict = {}
    for r_i in iterate_indices(need_compressed):
        for ix, sl in zip(need_compressed, r_i):
            slice_dict[ix] = sl
        chunk = np.empty(2**len(result_chunk_ixs), dtype=B.dtype)
        chunk = chunk.reshape(*(v.size for v in result_chunk_ixs))
        for irm in iterate_indices(remove_compress):
            for i, ival in zip(remove_compress, irm):
                slice_dict[i] = ival#slice(ival, ival+1)
            chunk_view = chunk[tuple(
                slice_dict.get(i, slice(None)) for i in result_chunk_ixs
            )]
            A_slice = A[slice_dict]
            B_slice = B[slice_dict]

            C_ixs = [v for v in result_chunk_ixs if v not in exist_compressed]
            C = Tensor('tmp', indices=C_ixs, data=chunk_view)
            contract_two_tensors(A_slice, B_slice, C)
        R.set_chunk(r_i, chunk)
    return R
