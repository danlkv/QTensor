from qtree import np_framework
from pyrofiler import Profiler

class BucketBackend:
    def process_bucket(self, bucket, no_sum=False):
        raise NotImplementedError

    def get_sliced_buckets(buckets, data_dict, slice_dict):
        raise NotImplementedError

class NumpyBackend(BucketBackend):
    def process_bucket(self, bucket, no_sum=False):
        return np_framework.process_bucket_np(bucket, no_sum=no_sum)

    def get_sliced_buckets(buckets, data_dict, slice_dict):
        return np_framework.get_sliced_buckets(buckets, data_dict, slice_dict)

class PerfBackend(BucketBackend):
    Backend = BucketBackend

    def __init__(self, *args, **kwargs):
        self.backend = self.Backend(*args, **kwargs)
        self._profiler = Profiler()
        self._profiler.cb = self._profile_callback

    def _profile_callback(self, result, label, indices):
        print(f"PROF:: perf data {label}: {result}")
        self._profile_results[indices] = result

    def process_bucket(self, bucket, no_sum=False):
        indices = [tensor.indices for tensor in bucket]
        with self._profiler.timing('process bucket time', indices):
            return self.backend.process_bucket(bucket, no_sum=no_sum)

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return self.backend.get_sliced_buckets(buckets, data_dict, slice_dict)


class PerfNumpyBackend(PerfBackend):
    Backend = NumpyBackend
