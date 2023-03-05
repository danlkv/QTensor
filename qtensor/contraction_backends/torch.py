from qtensor.tools.lazy_import import torch
import qtree
import numpy as np
from qtree import np_framework
from qtensor.contraction_backends import ContractionBackend
from .common import slice_numpy_tensor, get_einsum_expr
import string
CHARS = string.ascii_lowercase + string.ascii_uppercase

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

def get_einsum_expr_bucket(bucket, all_indices_list, result_indices):
    # converting elements to int will make stuff faster, 
    # but will drop support for char indices
    # all_indices_list = [int(x) for x in all_indices]
    # to_small_int = lambda x: all_indices_list.index(int(x))
    to_small_int = lambda x: all_indices_list.index(x)
    expr = ','.join(
        ''.join(CHARS[to_small_int(i)] for i in t.indices)
        for t in bucket) +\
            '->'+''.join(CHARS[to_small_int(i)] for i in result_indices)
    return expr





class TorchBackend(ContractionBackend):
    def __init__(self, device='cpu'):
        self.device = device
        self.dtype = ['float', 'double', 'complex64', 'complex128']
        self.width_dict = [set() for i in range(30)]
        self.width_bc = [[0,0] for i in range(30)] #(#distinct_bc, #bc)

    def process_bucket(self, bucket, no_sum=False):
        bucket.sort(key = lambda x: len(x.indices))
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        width = len(set(bucket[0].indices))

        for tensor in bucket[1:-1]:

            expr = get_einsum_expr(
                list(map(int, result_indices)), list(map(int, tensor.indices))
            )

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

        if len(bucket)>1:
            tensor = bucket[-1]
            expr = get_einsum_expr(
                list(map(int, result_indices)), list(map(int, tensor.indices))
                , contract = 1
            )
            result_data = torch.einsum(expr, result_data, tensor.data)
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int, reverse=True
            ))
        else:
            result_data = result_data.sum(axis=-1)



        if len(result_indices) > 0:
            first_index = result_indices[-1]
            result_indices = result_indices[:-1]
            tag = first_index.identity
        else:
            tag = 'f'
            result_indices = []

        # reduce
        result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                            data=result_data)
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
        
        expr = get_einsum_expr_bucket(bucket, all_indices_list, result_indices)
        expect = len(result_indices)
        result_data = torch.einsum(expr, *tensors)

        if len(result_indices) > 0:
            first_index, *_ = result_indices
            tag = str(first_index)
        else:
            tag = 'f'

        result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                            data=result_data)
        
        return result

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        sliced_buckets = []
        for bucket in buckets:
            sliced_bucket = []
            for tensor in bucket:
                # get data
                # sort tensor dimensions
                out_indices = list(sorted(tensor.indices, key=int, reverse=True))
                data = data_dict[tensor.data_key]
                # Works for torch tensors just fine
                data, new_indices = slice_numpy_tensor(data, tensor.indices, out_indices, slice_dict)

                if not isinstance(data, torch.Tensor):             
                    if self.device == 'gpu' and torch.cuda.is_available():
                        cuda = torch.device('cuda')
                        data = torch.from_numpy(data.astype(np.complex128)).to(cuda)
                    else:
                        data = torch.from_numpy(data.astype(np.complex128))
                else:
                    data = data.type(torch.complex128)
                # slice data
                sliced_bucket.append(
                    tensor.copy(indices=new_indices, data=data))
            sliced_buckets.append(sliced_bucket)

        return sliced_buckets

    def get_result_data(self, result):
        return np.transpose(result.data)
