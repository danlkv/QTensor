from qtree import np_framework

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
