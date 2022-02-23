from qtensor.tools.lazy_import import torch
import qtree
import numpy as np
from qtree import np_framework
from qtensor.contraction_backends import ContractionBackend
from qtensor.contraction_backends.numpy import get_einsum_expr
import pyrofiler
import time




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


class TorchEmbeddedBackend(ContractionBackend):
    def __init__(self, device = "gpu"):
        self.device = device
        self.dtype = ['float', 'double', 'complex64', 'complex128']
        self.width_dict = [set() for i in range(30)]
        self.width_bc = [[0,0] for i in range(30)] #(#distinct_bc, #bc)
        self.exprs = {}


    def process_bucket(self, bucket, bucket_index = 0, no_sum=False):

        result_indices = bucket[0].indices
        result_data = bucket[0].data
        
        if not isinstance(result_data, torch.Tensor):
            #print("Encountering: ",type(result_data))           
            if self.device == 'gpu' and torch.cuda.is_available():
                cuda = torch.device('cuda')
                result_data = torch.from_numpy(result_data).to(cuda)
            else:
                result_data = torch.from_numpy(result_data)

        for t,tensor in enumerate(bucket[1:]):

            '''
            Entry Point 1
            '''
            with pyrofiler.PROF.timing('Get Einsum Expr', reference=(bucket_index,t)):
                expr = qtree.utils.get_einsum_expr(
                    list(map(int, result_indices)), list(map(int, tensor.indices))
                )

            # @pyrofiler.PROF.cpu(desc='Get Einsum Expr', reference=(bucket_index,t))
            # def get_expr(result_indices, tensor):
            #     return qtree.utils.get_einsum_expr(
            #         list(map(int, result_indices)), list(map(int, tensor.indices))
            #     )
            
            # expr = get_expr(result_indices, tensor)

            '''
            Entry Point 2
            '''

            with pyrofiler.PROF.timing('Data Type Adjust', reference=(bucket_index,t)):
                if not isinstance(tensor._data, torch.Tensor):             
                    if self.device == 'gpu' and torch.cuda.is_available():
                        cuda = torch.device('cuda')
                        tensor._data = torch.from_numpy(tensor._data).to(cuda)
                    else:
                        tensor._data = torch.from_numpy(tensor._data)


            # @pyrofiler.PROF.cpu(desc='Data Type Adjust', reference=(bucket_index,t))
            # def adjustType(tensor):
            #     if not isinstance(tensor._data, torch.Tensor):             
            #         if self.device == 'gpu' and torch.cuda.is_available():
            #             cuda = torch.device('cuda')
            #             tensor._data = torch.from_numpy(tensor._data).to(cuda)
            #         else:
            #             tensor._data = torch.from_numpy(tensor._data)
            # adjustType(tensor)

            '''
            Entry Point 3
            '''
            with pyrofiler.PROF.timing('Device Transport', reference=(bucket_index,t)):
                if self.device == 'gpu':
                    if result_data.device != "gpu":
                        result_data = result_data.to(torch.device('cuda'))
                    if tensor.data.device != "gpu":
                        tensor._data = tensor._data.to(torch.device("cuda"))
                else:
                    if result_data.device != "cpu":
                        result_data = result_data.cpu()
                    if tensor.data.device != "cpu":
                        tensor._data = tensor._data.cpu()
            # @pyrofiler.PROF.cpu(desc="Device Transport", reference=(bucket_index,t))            
            # def transportData(tensor, result_data):
            #     if self.device == 'gpu':
            #         if result_data.device != "gpu":
            #             result_data = result_data.to(torch.device('cuda'))
            #         if tensor.data.device != "gpu":
            #             tensor._data = tensor._data.to(torch.device("cuda"))
            #     else:
            #         if result_data.device != "cpu":
            #             result_data = result_data.cpu()
            #         if tensor.data.device != "cpu":
            #             tensor._data = tensor._data.cpu()
            
            # transportData(tensor, result_data)

            '''
            Entry Point 4
            '''
            with pyrofiler.PROF.timing('Einsum Compute', reference=(bucket_index,t)):
                result_data = torch.einsum(expr, result_data, tensor.data)


            # @pyrofiler.PROF.cpu('Einsum Compute', reference=(bucket_index,t))
            # def computeEinsum(expr, result_data, tensor):
            #     return torch.einsum(expr, result_data, tensor.data)

            # result_data = computeEinsum(expr, result_data, tensor)

            '''
            Entry Point 5
            '''

            with pyrofiler.PROF.timing('Result Indices', reference=(bucket_index,t)):
                result_indices = tuple(sorted(
                    set(result_indices + tensor.indices),
                    key=int)
                )

            # @pyrofiler.PROF.cpu('Result Indices', reference=(bucket_index,t))
            # def getResultIndices(result_indices, tensor):
            #     return tuple(sorted(
            #         set(result_indices + tensor.indices),
            #         key=int)
            #     )
            
            # result_indices = getResultIndices(result_indices, tensor)

        if len(result_indices) > 0:
            if not no_sum:  # trim first index
                first_index, *result_indices = result_indices
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
                                data=torch.sum(result_data, axis=0))
        return result

    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        result_indices = bucket[0].indices
        # print("result_indices", result_indices)
        result_data = bucket[0].data

        if not isinstance(result_data, torch.Tensor):
            #print("Encountering: ",type(result_data))           
            if self.device == 'gpu' and torch.cuda.is_available():
                cuda = torch.device('cuda')
                result_data = torch.from_numpy(result_data).to(cuda)
            else:
                result_data = torch.from_numpy(result_data)




        all_indices = set(sum((list(t.indices) for t in bucket), []))
        result_indices = all_indices - set(ixs)
        all_indices_list = list(all_indices)
        to_small_int = lambda x: all_indices_list.index(x)
        tensors = []
        is128 = False
        for tensor in bucket:

            if not isinstance(tensor._data, torch.Tensor):             
                if self.device == 'gpu' and torch.cuda.is_available():
                    cuda = torch.device('cuda')
                    tensor._data = torch.from_numpy(tensor._data).to(cuda)
                else:
                    tensor._data = torch.from_numpy(tensor._data)
            
            if self.device == 'gpu':
                if result_data.device != "gpu":
                    result_data = result_data.to(torch.device('cuda'))
                if tensor.data.device != "gpu":
                    tensor._data = tensor._data.to(torch.device("cuda"))
            else:
                if result_data.device != "cpu":
                    result_data = result_data.cpu()
                if tensor.data.device != "cpu":
                    tensor._data = tensor._data.cpu()

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
                transpose_order = np.argsort(list(map(int, tensor.indices)))
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
        try:
            return result.data.cpu()
        except:
            return result.data