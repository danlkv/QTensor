import torch
from qtree import np_framework

from qtensor.contraction_backends import ContractionBackend

class TorchBackend(ContractionBackend):
    def process_bucket(self, bucket, no_sum=False):
        raise NotImplementedError

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        np_buckets = np_framework.get_sliced_np_buckets(buckets, data_dict, slice_dict)
        torch_buckets = [
            [torch.Tensor(x) for x in bucket]
            for bucket in np_buckets
        ]
        return torch_buckets

    def get_result_data(self, result):
        raise NotImplementedError

def test_torch_backend():
    print('Hello world')
    assert False

