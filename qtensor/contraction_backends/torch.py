from qtensor.tools.lazy_import import torch
import qtree
import numpy as np
from qtree import np_framework
from qtensor.contraction_backends import ContractionBackend
from qtensor.contraction_backends.numpy import get_einsum_expr
def qtree2torch_tensor(tensor, data_dict):
    """ Converts qtree tensor to pytorch tensor using data dict"""
    if isinstance(tensor.data, torch.Tensor):
        return tensor
    if tensor.data is not None:
        data = tensor.data
    else:
        data = data_dict[tensor.data_key]
    torch_t = torch.from_numpy(data)
    data_dict[tensor.data_key] = torch_t
    return tensor.copy(data=torch_t)


def get_einsum_expr(idx1, idx2):
    """
    Takes two tuples of indices and returns an einsum expression
    to evaluate the sum over repeating indices

    Parameters
    ----------
    idx1 : list-like
          indices of the first argument
    idx2 : list-like
          indices of the second argument

    Returns
    -------
    expr : str
          Einsum command to sum over indices repeating in idx1
          and idx2.
    """
    result_indices = sorted(list(set(idx1 + idx2)), reverse=True)
    # remap indices to reduce their order, as einsum does not like
    # large numbers
    idx_to_least_idx = {old_idx: new_idx for new_idx, old_idx
                        in enumerate(result_indices)}

    str1 = ''.join(qtree.utils.num_to_alpha(idx_to_least_idx[ii]) for ii in idx1)
    str2 = ''.join(qtree.utils.num_to_alpha(idx_to_least_idx[ii]) for ii in idx2)
    str3 = ''.join(qtree.utils.num_to_alpha(idx_to_least_idx[ii]) for ii in result_indices)
    return str1 + ',' + str2 + '->' + str3



class TorchBackend(ContractionBackend):
    def __init__(self, device='cpu'):
        self.device = device
        self.dtype = ['float', 'double', 'complex64', 'complex128']
        self.width_dict = [set() for i in range(30)]
        self.width_bc = [[0,0] for i in range(30)] #(#distinct_bc, #bc)
        self.exprs = {}


    def process_bucket(self, bucket, no_sum=False):
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        width = len(set(bucket[0].indices))
        #print("w:",width)

        for tensor in bucket[1:]:

            expr = get_einsum_expr(
                list(map(int, result_indices)), list(map(int, tensor.indices))
            )

            if expr not in self.exprs.keys():
                self.exprs[expr] = 1
            else:
                self.exprs[expr] += 1

            result_data = torch.einsum(expr, result_data, tensor.data)

            # Merge and sort indices and shapes
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int, reverse=True
            )
            )
            
            size = len(set(tensor.indices))
            if size > width:
                width = size

            self.width_dict[width].add(expr)
            self.width_bc[width][0] = len(self.width_dict[width])
            self.width_bc[width][1] += 1

        if len(result_indices) > 0:
            if not no_sum:  # trim first index
                first_index = result_indices[-1]
                result_indices = result_indices[:-1]
            else:
                first_index, *_ = result_indices
            tag = first_index.identity
        else:
            tag = 'f'
            result_indices = []

        # reduce
        if no_sum:
            result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                                data=result_data)
        else:
            result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                                data=torch.sum(result_data, axis=-1))
        
        #print("summary:",sorted(self.exprs.items(), key=lambda x: x[1], reverse=True))
        #print("stats:",self.width_bc)
        return result

    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        result_indices = bucket[0].indices
        # print("result_indices", result_indices)
        result_data = bucket[0].data
        all_indices = set(sum((list(t.indices) for t in bucket), []))
        result_indices = all_indices - set(ixs)
        all_indices_list = list(all_indices)
        to_small_int = lambda x: all_indices_list.index(x)
        tensors = []
        is128 = False
        for tensor in bucket:
            if tensor.data.dtype in [torch.float64]:
                tensors.append(tensor.data.type(torch.complex64))
            else:
                tensors.append(tensor.data)
            if tensor.data.dtype == torch.complex128:
                is128 = True
        
        if is128:
            for i in range(len(tensors)):
                tensors[i] = tensors[i].type(torch.complex128)
        
        expr = get_einsum_expr(bucket, all_indices_list, result_indices)
        # print("expr:", expr)
        if expr not in self.exprs.keys():
            self.exprs[expr] = 1
        else:
            self.exprs[expr] += 1

        expect = len(result_indices)
        result_data = torch.einsum(expr, *tensors)

        if len(result_indices) > 0:
            first_index, *_ = result_indices
            tag = str(first_index)
        else:
            tag = 'f'

        result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                            data=result_data)
        
        # print("summary:",sorted(self.exprs.items(), key=lambda x: x[1], reverse=True))
        # print("# distinct buckets:", len(self.exprs))
        return result

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        sliced_buckets = []
        for bucket in buckets:
            sliced_bucket = []
            for tensor in bucket:
                # get data
                # sort tensor dimensions
                transpose_order = np.argsort(list(map(int, tensor.indices)))[::-1]
                data = data_dict[tensor.data_key]
                if not isinstance(data, torch.Tensor):             
                    if self.device == 'gpu' and torch.cuda.is_available():
                        cuda = torch.device('cuda')
                        data = torch.from_numpy(data).to(cuda)
                    else:
                        data = torch.from_numpy(data)

                data = data.permute(tuple(transpose_order))
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

                sliced_bucket.append(
                    tensor.copy(indices=indices_sliced, data=data))
            sliced_buckets.append(sliced_bucket)

        return sliced_buckets

    def get_result_data(self, result):
        return result.data
