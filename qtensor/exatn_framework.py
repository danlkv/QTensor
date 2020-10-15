"""
This file implements Numpy framework of the
simulator. It's main use is in conjunction with the :py:mod:`optimizer`
module, and example programs are listed in :py:mod:`simulator` module.
"""

import lazy_import
import numpy as np

import qtree.utils as utils

exatn = lazy_import.lazy_module('exatn')

from collections import namedtuple

TensorInfo = namedtuple("TensorInfo", "name indices")

def get_sliced_exatn_buckets(buckets, data_dict, slice_dict):
    """
    Takes placeholder buckets and populates them with
    actual sliced values. This function is a sum of
    :func:`get_np_buckets` and :func:`slice_np_buckets`

    Parameters
    ----------
    buckets : list of list
              buckets as returned by :py:meth:`circ2buckets`
              and :py:meth:`reorder_buckets`.
    data_dict : dict
              dictionary containing values for the placeholder Tensors
    slice_dict : dict
              Current subtensor along the sliced variables
              in the form {variable: slice}
    Returns
    -------
    sliced_buckets : list of lists
              buckets with sliced Numpy tensors
    """
    # import pdb
    # pdb.set_trace()

    # Create np buckets from buckets
    sliced_buckets = []
    for bucket in buckets:
        sliced_bucket = []
        for tensor in bucket:
            # get data
            # sort tensor dimensions
            transpose_order = np.argsort(list(map(int, tensor.indices)))
            data = np.transpose(data_dict[tensor.data_key],
                                transpose_order)
            # transpose indices
            indices_sorted = [tensor.indices[pp]
                              for pp in transpose_order]

            # slice data
            slice_bounds = []
            for idx in indices_sorted:
                try:
                    slice_bounds.append(slice_dict[idx])
                except KeyError:
                    slice_bounds.append(slice(None))

            data = data[tuple(slice_bounds)]

            # update indices
            indices_sliced = [idx.copy(size=size) for idx, size in
                              zip(indices_sorted, data.shape)]
            indices_sliced = [i for sl, i in zip(slice_bounds, indices_sliced) if not isinstance(sl, int)]
            assert len(data.shape) == len(indices_sliced)

            print(f"creating {tensor.name}")
            exatn.createTensor(tensor.name, data)

            sliced_bucket.append(TensorInfo(tensor.name, indices_sliced))
        sliced_buckets.append(sliced_bucket)

    return sliced_buckets

def idx_to_string(idx):
    idx = map(int, idx)
    letters = list(map(utils.num_to_alpha, idx))
    return ",".join(letters)

def tensor_to_string(tensor):
    print(tensor.indices)
    idx = idx_to_string(tensor.indices)
    return tensor.name + "(" + idx + ")"

def get_exatn_expr(tensor1, tensor2, result_name, result_idx):
    # remap indices to reduce their order, as einsum does not like
    # large numbers
    all_indices = set.union(set(tensor1.indices), set(tensor2.indices))
    idx_to_least_idx = {old_idx: new_idx for new_idx, old_idx
                        in enumerate(all_indices)}
    tensor1 = TensorInfo(name=tensor1.name, indices=[idx_to_least_idx[idx] for idx in tensor1.indices])
    tensor2 = TensorInfo(name=tensor2.name, indices=[idx_to_least_idx[idx] for idx in tensor2.indices])
    result_idx = [idx_to_least_idx[idx] for idx in result_idx]

    # T(a,b,c) = A(a,b) * B(b,c)
    str1 = tensor_to_string(tensor1)
    str2 = tensor_to_string(tensor2)
    str3 = f"{result_name}({idx_to_string(result_idx)})"

    return f"{str3} = {str2} * {str1}"

def get_result_indices(idx1, idx2, contract=True):
    result_indices = tuple(sorted(set(idx1 + idx2), key=int))
    if contract:
        result_indices = result_indices[1:]
    return result_indices
   

def process_bucket_exatn(bucket, no_sum=False, result_id=0):
    """
    Process bucket in the bucket elimination algorithm.
    We multiply all tensors in the bucket and sum over the
    variable which the bucket corresponds to. This way the
    variable of the bucket is removed from the expression.

    Parameters
    ----------
    bucket : list
           List containing tuples of tensors (gates) with their indices.

    no_sum : bool
           If no summation should be done over the buckets's variable

    Returns
    -------
    tensor : optimizer.Tensor
           wrapper tensor object holding the result
    """

    pr_info = bucket[0]
    n = len(bucket)

    tmp_id = 0

    for i, t_info in enumerate(bucket[1:]):
        no_hcon = n == 2 or i == n - 1 # TODO better check if hypercontraction is required
        result_indices = get_result_indices(pr_info.indices, t_info.indices, contract=no_hcon)
        if no_hcon:
            no_sum = True
        else:
            # raise Exception('QTensorError: Exatn Hyper-contractions are not supported at the moment')
            no_sum = False

        new_name = f"C{np.random.randint(0, 1000000000)}"
        exatn.createTensor(new_name, np.empty([2]*len(result_indices), dtype=complex))
        expr = get_exatn_expr(pr_info, t_info, new_name, result_indices)

        pr_info = TensorInfo(new_name, result_indices)
        exatn.contractTensors(expr)

    return pr_info
