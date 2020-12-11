from qtensor.contraction_backends import ContractionBackend, NumpyBackend
from qtensor.utils import ReportTable
from pyrofiler import timing

class PerfBackend(ContractionBackend):
    Backend = ContractionBackend

    def __init__(self, *args, print=False, num_lines=20, **kwargs):
        self.backend = self.Backend(*args, **kwargs)
        self._print = print
        self.max_lines = num_lines
        self._profile_results = {}
        self.report_table = ReportTable(measure=['max','mean','sum'], max_records=num_lines)

    def _profile_callback(self, time, label, indices):
        if self._print:
            print(f"PROF:: perf data {label}: {time}")
        self._profile_results[str(indices)] = indices, time

    @classmethod
    def from_backend(cls, backend, *args, **kwargs):
        """ Dynamically create and instantiate a class with a given backend. """
        class CustomGeneratedBackend(cls):
            Backend = backend
        return CustomGeneratedBackend(*args, **kwargs)

    def process_bucket(self, bucket, no_sum=False):
        indices = [tensor.indices for tensor in bucket]
        with timing('process bucket time', indices
                         , callback=self._profile_callback):
            return self.backend.process_bucket(bucket, no_sum=no_sum)

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return self.backend.get_sliced_buckets(buckets, data_dict, slice_dict)

    def get_result_data(self, result):
        return self.backend.get_result_data(result)

    def _perfect_bucket_flop(self, bucket_indices):
        resulting_indices = list(set.union(*[set(ixs) for ixs in bucket_indices]))
        # The first index is contracted
        resulting_indices = resulting_indices[1:]
        # don't take index size into account
        n_multiplications = len(bucket_indices)
        size_of_result = 2**len(resulting_indices)
        summation_index_size = 2
        n_summations = summation_index_size - 1
        op = size_of_result*( n_summations + n_multiplications )
        return op


    def gen_report(self):
        data = self._profile_results.values()
        # -- sotrt data with respect to time
        data = sorted(data, key= lambda pair: pair[1], reverse=True)
        # -- report on largest contractions
        max_lines = self.max_lines

        report_lines =  [str([i, ixs, time ]) for i, (ixs, time) in enumerate(data[:max_lines])]
        rep = '\n'.join(report_lines[:max_lines])
        if len(report_lines) > max_lines:
            rep += f'\n ... and {len(data)-max_lines} lines more...'

        # -- report on totals
        for indices, time in  data[:max_lines]:
            self.report_table.record(
                bucket_len = len(indices)
                , time = time
                , flop = self._perfect_bucket_flop(indices)
                , FLOPS = self._perfect_bucket_flop(indices)/time
                , max_size = max([len(ixs) for ixs in indices])
                , min_size = min([len(ixs) for ixs in indices])
                , result_size = len(set.union(*[set(i) for i in indices])) - 1
            )

        print(self.report_table.markdown())


        # -- report on totals
        total_data = len(data)
        total_time = sum(d[1] for d in data)
        rep += '\n======\n'
        rep += 'Total time: ' + str(total_time)
        rep += '\nTotal bucket contractions: ' + str(total_data)
        rep += '\nMean time for contraction: ' + str(total_time/total_data)
        rep += '\n'
        return rep

class PerfNumpyBackend(PerfBackend):
    Backend = NumpyBackend
