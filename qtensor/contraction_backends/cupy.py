import qtree
from qtensor.tools.lazy_import import cupy as cp
from qtensor.contraction_backends import ContractionBackend
#from qtensor.contraction_backends.numpy import get_einsum_expr
from .common import slice_numpy_tensor, get_einsum_expr


class CuPyBackend(ContractionBackend):
    
    # Replace all torch methods with cupy's analog
    
    def process_bucket(self, bucket, no_sum=False):
        bucket.sort(key = lambda x: len(x.indices))
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        for tensor in bucket[1:]:

            expr = get_einsum_expr(
                list(map(int, result_indices)), list(map(int, tensor.indices))
            )
            
            '''
            Change 1: Using cp.einsum not torch.einsum
            ''' 
            result_data = cp.einsum(expr, result_data, tensor.data)

            # Merge and sort indices and shapes
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int, reverse=True)
            )

        if len(result_indices) > 0:
            if not no_sum:  # trim first index
                contract_index = result_indices[-1]
                result_indices = result_indices[:-1]
            else:
                contract_index = result_indices[-1]
            tag = contract_index.identity
        else:
            tag = 'f'
            result_indices = []

        # reduce
        if no_sum:
            result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                                data=result_data)
        else:
            result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                                data=cp.sum(result_data, axis=-1))
        return result

    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        all_indices = set(sum((list(t.indices) for t in bucket), []))
        result_indices = all_indices - set(ixs)
        all_indices_list = list(all_indices)
        to_small_int = lambda x: all_indices_list.index(x)
        
        expr = get_einsum_expr(bucket, all_indices_list, result_indices)

        params = []
        for tensor in bucket:
            params.append(tensor.data)
            params.append(list(map(to_small_int, tensor.indices)))
        params.append(list(map(to_small_int, result_indices)))
        
        expect = len(result_indices)

        result_data = cp.einsum(*params)

        if len(result_indices) > 0:
            first_index, *_ = result_indices
            tag = str(first_index)
        else:
            tag = 'f'

        result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                            data=result_data)
        return result

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        sliced_buckets = []
        for bucket in buckets:
            sliced_bucket = []
            for tensor in bucket:
                # get data
                # sort tensor dimensions
                
                '''
                Change 2:
                Original: Using np to sort
                New:      Use cp.argsort() to sort it needs to be casted into an python list
                '''
                # transpose_order = np.argsort(list(map(int, tensor.indices)))
                # cp.argsort requires input to be cp array
                #print(tensor.indices)
                transpose_order = cp.argsort(cp.asarray(list(map(int, tensor.indices)))).tolist()
                transpose_order = list(reversed(transpose_order))
                
                '''
                Change 2:
                Original: Data is all converted into torch.tensor and use torch api, the results are in torch
                New:      Convert all data to CuPy.ndarray, will raise exceptional signal
                '''
                data = data_dict[tensor.data_key]
                try:
                    data = cp.asarray(data)
                    data = data.transpose(tuple(transpose_order))
                except:
                    print("CuPy Backend doesn't support gradient.")
                
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
        return cp.transpose(result.data)
