import numpy as np
from qtensor.contraction_backends import ContractionBackend, NumpyBackend
from pyrofiler import timing
from qtensor.tools.lazy_import import torch, pandas

class PerfBackend(ContractionBackend):
    Backend = ContractionBackend

    def __init__(self, *args, print=False, num_lines=20, **kwargs):
        self.backend = self.Backend(*args, **kwargs)
        self._print = print
        self.max_lines = num_lines
        self._profile_results = {}
        self.report_table = pandas.DataFrame(columns=['bucket_len', 'time', 'flop', 'FLOPS', 'max_size', 'min_size', 'result_size'])

    def _profile_callback(self, time, label, indices):
        if self._print:
            print(f"PROF:: perf data {label}: {time}")
        self._profile_results[str(id(indices))] = indices, time
        

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

    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        indices = [tensor.indices for tensor in bucket]
        with timing('process bucket time', indices
                         , callback=self._profile_callback):
            return self.backend.process_bucket_merged(ixs, bucket, no_sum=no_sum)

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return self.backend.get_sliced_buckets(buckets, data_dict, slice_dict)

    def get_result_data(self, result):
        return self.backend.get_result_data(result)

    def _pairwise_flop_mem(self, indices, contract_last=0):
        """
        Args:
            indices: list of two index lists
        Returns:
            next_indices: list of indices after contraction
            tuple(flops, mem): required resources for contraction
        """
        next_indices = list(set().union(*indices))
        next_indices.sort(key=int, reverse=True)
        flop = np.prod([i.size for i in next_indices])
        next_indices = next_indices[:-contract_last]
        mem = 0
        for ilist in [next_indices]+indices:
            mem += np.prod([i.size for i in ilist])
        return next_indices, (flop, mem)

    def _perfect_bucket_flop_mem(self, bucket_indices, show = False):
        """
        Returns estimation of flops for a bucket
        """
        bucket_indices = sorted(bucket_indices, key=lambda x: len(x))
        flop_list = []
        mem_list = []
        accum = bucket_indices[0]
        for ixs in bucket_indices[1:1]:
            indices = [accum, ixs]
            # -- Get pairwise contraction flops
            accum, (flop, mem) = self._pairwise_flop_mem(indices)
            # --
            flop_list.append(flop)
            mem_list.append(mem)
            accum = next_indices
        # -- last contraction removes the smallest index
        indices = [accum, bucket_indices[-1]]
        _, (flop, mem) = self._pairwise_flop_mem(indices, contract_last=1)
        # --
        flop_list.append(flop)
        mem_list.append(mem)

        return sum(flop_list), max(mem_list)



    def gen_report(self, show = True):
        data = self._profile_results.values()
        # -- sotrt data with respect to time
        #data = sorted(data, key= lambda pair: pair[1], reverse=True)
        data = list(data)
        # -- report on largest contractions
        max_lines = self.max_lines

        df = pandas.DataFrame(data, columns=['indices', 'time'])

        df.sort_values(by='time', ascending=False, inplace=True)
        rep = df.head(max_lines).to_string()
        if len(data) > max_lines:
            rep += f'\n ... and {len(data)-max_lines} lines more...'

        # -- report on totals
        # max_line should not be inolved for recording

        report_data = {
            'bucket_len': [],
            'time': [],
            'flop': [],
            'FLOPS': [],
            'max_size': [],
            'min_size': [],
            'result_size': [],
        }
        for indices, time in  data:
            max_size = max([len(i) for i in indices])
            min_size = min([len(i) for i in indices])
            flop, mem = self._perfect_bucket_flop_mem(indices)
            result_size = len(set.union(*[set(i) for i in indices])) - 1
            report_data['bucket_len'].append(len(indices))
            report_data['time'].append(time)
            report_data['flop'].append(flop)
            report_data['FLOPS'].append(flop/time)
            report_data['max_size'].append(max_size)
            report_data['min_size'].append(min_size)
            report_data['result_size'].append(result_size)

        self.report_table = pandas.DataFrame(report_data)
        if show:
            print(self.report_table.to_markdown())


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


class GPUPerfBackend(PerfBackend):
    def process_bucket(self, bucket, no_sum=False):
        indices = [tensor.indices for tensor in bucket]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        out = self.backend.process_bucket(bucket, no_sum=no_sum)
        
        end.record()
        torch.cuda.synchronize()
        time= start.elapsed_time(end)/1000

        # sorted(self.backend.exprs.items(), key=lambda x: x[1], reverse=True)
        # print("summary:",sorted(self.backend.exprs.items(), key=lambda x: x[1], reverse=True))

        self._profile_callback(time,'process bucket time',indices)
        return out
