from qtree import np_framework
from pyrofiler import timing
# from time import time


# def printer(result, description='Profile results'):
#     print(description, ':', result)
#
#
# def timing(description: str, callback=printer) -> None:
#     start = time()
#     yield
#     ellapsed_time = time() - start
#     callback(ellapsed_time, description=description)


class BucketBackend:
    def process_bucket(self, bucket, no_sum=False):
        raise NotImplementedError

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        raise NotImplementedError

class NumpyBackend(BucketBackend):
    def process_bucket(self, bucket, no_sum=False):
        res =  np_framework.process_bucket_np(bucket, no_sum=no_sum)
        return res

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return np_framework.get_sliced_np_buckets(buckets, data_dict, slice_dict)

class PerfBackend(BucketBackend):
    Backend = BucketBackend

    def __init__(self, *args, print=False, num_lines=10, **kwargs):
        self.backend = self.Backend(*args, **kwargs)
        self._print = print
        self.max_lines = num_lines
        self._profile_results = {}

    def _profile_callback(self, elapsed_time, label, indices):
        if self._print:
            print(f"PROF:: perf data {label}: {elapsed_time}")
        self._profile_results[str(indices)] = indices, elapsed_time

    def process_bucket(self, bucket, no_sum=False):
        indices = [tensor.indices for tensor in bucket]
        with timing('process bucket time'
                , indices
                         , callback=self._profile_callback):
            return self.backend.process_bucket(bucket, no_sum=no_sum)

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return self.backend.get_sliced_buckets(buckets, data_dict, slice_dict)

    def gen_report(self):
        data = self._profile_results.values()
        data = sorted(data, key= lambda pair: pair[1], reverse=True)
        report_lines =  [str([ d[1], d[0] ]) for d in data]
        max_lines = self.max_lines
        total_data = len(data)
        total_time = sum(d[1] for d in data)
        rep = ""
        # rep = '\n'.join(report_lines[:max_lines])
        # if len(report_lines) > max_lines:
        #     rep += f'\n ... and {total_data-max_lines} lines more...'
        # rep += '\n======\n'
        # rep += 'Total time: ' + str(total_time)
        # rep += '\nTotal bucket contractions: ' + str(total_data)
        # rep += '\nMean time for a contraction: ' + str(total_time/total_data)
        # rep += '\n'
        # rep += '\n======\n'
        rep += 'Total_time: ' + str(total_time)
        rep += '\nNum_bucket_contractions: ' + str(total_data)
        rep += '\nMean_time_for_contraction: ' + str(total_time / total_data)
        rep += '\n'
        # rep += '\n======\n'
        return rep




class PerfNumpyBackend(PerfBackend):
    Backend = NumpyBackend
    backend = Backend