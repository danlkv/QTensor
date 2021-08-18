from qtensor.tools.lazy_import import torch
import qtree
import numpy as np
from qtensor.contraction_backends import ContractionBackend
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



'bucket = [t.indices for t in bucket] for converting a tensor' 

def bucketWidth(bucket):
    """ Returns the width of a bucket"""
    return len(max(bucket, key = len))



class TorchMixBackend(ContractionBackend):

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()


    def process_bucket(self, bucket, no_sum=False):
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        for tensor in bucket[1:]:

            expr = qtree.utils.get_einsum_expr(
                list(map(int, result_indices)), list(map(int, tensor.indices))
            )

            result_data = torch.einsum(expr, result_data, tensor.data)

            # Merge and sort indices and shapes
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int)
            )

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

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        sliced_buckets = []
        for bucket in buckets:
            sliced_bucket = []
            bucket_width = bucketWidth(bucket)
            for tensor in bucket:
                transpose_order = np.argsort(list(map(int, tensor.indices)))
                data = data_dict[tensor.data_key]
                if not isinstance(data, torch.Tensor):
                    if self.cuda_available:
                        if bucket_width > 8:
                            cuda = torch.device('cuda')
                            data = torch.from_numpy(data).to(cuda)
                        else:
                            data = torch.from_numpy(data)
                    else:
                        raise Exception("cuda is not available on this machine")
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
