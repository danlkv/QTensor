from qtensor.tools.lazy_import import torch
import qtree
from qtree import np_framework

from qtensor.contraction_backends import ContractionBackend
def qtree2torch_tensor(tensor, data_dict):
    """ Converts qtree tensor to pytorch tensor using data dict"""
    if tensor.data is not None:
        data = tensor.data
    else:
        data = data_dict[tensor.data_key]
    torch_t = torch.from_numpy(data)
    data_dict[tensor.data_key] = torch_t
    return tensor.copy(data=torch_t)


class TorchBackend(ContractionBackend):
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
        np_buckets = np_framework.get_sliced_np_buckets(buckets, data_dict, slice_dict)
        torch_buckets = [
            [qtree2torch_tensor(x, data_dict) for x in bucket]
            for bucket in np_buckets
        ]
        return torch_buckets

    def get_result_data(self, result):
        return result.data
