from qtree import np_framework
from qtree import optimizer as opt

from pyrofiler import timing
from qensor.utils import ReportTable
import numpy as np

import tcontract

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

class ExaTnBackend(BucketBackend):
    def process_bucket(self, bucket, no_sum=False):
        res =  process_bucket_exatn(bucket, no_sum=no_sum)
        return res

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return get_sliced_exatn_buckets(buckets, data_dict, slice_dict)

class CMKLExtendedBackend(BucketBackend):
    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return np_framework.get_sliced_np_buckets(buckets, data_dict, slice_dict)

    def process_bucket(self, bucket, no_sum=False):
        result_indices = bucket[0].indices
        result_data = bucket[0].data

        for tensor in bucket[1:]:
            """
            next_result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int)
            )
            ixc = list(map(int, next_result_indices))
            idx_to_least_idx = {old_idx: new_idx for new_idx, old_idx
                            in enumerate(ixc)}

            ixa, ixb = list(map(int, result_indices)), list(map(int, tensor.indices))
            ixa, ixb = list(map(lambda x: idx_to_least_idx[x], ixa)), list(map(lambda x: idx_to_least_idx[x], ixb))
            print(len(ixa), len(ixb), len(ixc))
            #print(result_data.shape, len(ixb), len(ixc))
            ixc = list(map(lambda x: idx_to_least_idx[x], ixc))

            result_data = np.einsum(result_data, ixa, tensor.data, ixb, ixc)
            result_indices = next_result_indices
            """
            ixa, ixb = result_indices, tensor.indices
            common_ids = sorted(list(set.intersection(set(ixa), set(ixb))), key=int)
            distinct_a = [x for x in sorted(ixa, key=int) if x not in common_ids]
            distinct_b = [x for x in sorted(ixb, key=int) if x not in common_ids]
            transp_a = [ixa.index(x) for x in common_ids+distinct_a]
            transp_b = [ixb.index(x) for x in common_ids+distinct_b]
            a = result_data.transpose(transp_a)
            b = tensor.data.transpose(transp_b)
            n, m, k = 2**len(common_ids), 2**len(distinct_a), 2**len(distinct_b)
            a = a.reshape(n, m)
            b = b.reshape(n, k)

            c = np.empty((n, m, k), dtype=np.complex128)
            tcontract.mkl_contract_complex(a, b, c)

            # Merge and sort indices and shapes
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int)
            )
            result_data = c.reshape([2 for _ in result_indices])

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
            result = opt.Tensor(f'E{tag}', result_indices,
                                data=result_data)
        else:
            result = opt.Tensor(f'E{tag}', result_indices,
                                data=np.sum(result_data, axis=0))
        return result

class PerfBackend(BucketBackend):
    Backend = BucketBackend

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


    def process_bucket(self, bucket, no_sum=False):
        indices = [tensor.indices for tensor in bucket]
        with timing('process bucket time', indices
                         , callback=self._profile_callback):
            return self.backend.process_bucket(bucket, no_sum=no_sum)

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return self.backend.get_sliced_buckets(buckets, data_dict, slice_dict)

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
            kwargs= dict(
                bucket_len = len(indices)
                , time = time
                , flop = self._perfect_bucket_flop(indices)
                , FLOPS = self._perfect_bucket_flop(indices)/time
                , max_size = max([len(ixs) for ixs in indices])
                , min_size = min([len(ixs) for ixs in indices])
                , result_size = len(set.union(*[set(i) for i in indices])) - 1
            )
            self.report_table.record( **kwargs)

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
