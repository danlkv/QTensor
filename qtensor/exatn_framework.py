"""
This file implements Numpy framework of the
simulator. It's main use is in conjunction with the :py:mod:`optimizer`
module, and example programs are listed in :py:mod:`simulator` module.
"""

import numpy as np
import copy
import qtree.operators as ops
import qtree.optimizer as opt
import qtree.utils as utils
import exatn

from collections import namedtuple

TensorInfo = namedtuple("TensorInfo", "name indices")

def idx_to_string(idx):
    idx = map(int, idx)
    letters = list(map(utils.num_to_alpha, idx))
    return ",".join(letters)

def tensor_to_string(tensor):
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

    return f"{str1} = {str2} * {str3}"

def process_bucket_exatn(bucket, no_sum=False):
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
    print("BBUCKET")
    for x in bucket:
        exatn.createTensor(x.name, x.data.astype(complex))
    print("ABUCKET")
    prev_result = bucket[0]
    pr_info = TensorInfo(prev_result.name, prev_result.indices)

    
    for i, tensor in enumerate(bucket[1:]):
        t_info = TensorInfo(tensor.name, tensor.indices)

        result_indices = tuple(sorted(
            set(pr_info.indices + t_info.indices),
            key=int)
        )
        if len(bucket) == 2:
            result_indices = result_indices[1:]
            no_sum = True

        new_name = f"C{np.random.randint(0, 100000)}"
        exatn.createTensor(new_name)
        expr = get_exatn_expr(pr_info, t_info, new_name, result_indices)
        
        pr_info = TensorInfo(new_name, result_indices)
        print("BCONTRACT")
        print(expr)
        exatn.contractTensors(expr)
        print("ACONTRACT")

    result_indices = pr_info.indices


    if len(result_indices) > 0:
        if not no_sum:  # trim first index
            first_index, *result_indices = result_indices
        else:
            first_index, *_ = result_indices
        tag = first_index.identity
    else:
        tag = 'f'
        result_indices = []

    print("BRESULT")
    result_data = exatn.getLocalTensor(pr_info.name)
    print("ARESULT")

    # reduce
    if no_sum:
        result = opt.Tensor(f'E{tag}', result_indices, data=result_data)
    else:
        result = opt.Tensor(f'E{tag}', result_indices, 
                                data=np.sum(result_data, axis=0))
    return result
