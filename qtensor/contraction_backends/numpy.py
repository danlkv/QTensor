from qtensor.contraction_backends import ContractionBackend
import qtree.optimizer as opt
from qtree import np_framework
import opt_einsum
import numpy as np
import string

CHARS = string.ascii_lowercase + string.ascii_uppercase

class NumpyBackend(ContractionBackend):
    def __init__(self):
        super().__init__()
        #self.pbar = tqdm(desc='Buckets', position=2)
        #self.status_bar = tqdm(desc='Current status', position=3, bar_format='{desc}')

    def process_bucket(self, bucket, no_sum=False):
        res = np_framework.process_bucket_np(bucket, no_sum=no_sum)
        return res

    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        all_indices = set(sum((list(t.indices) for t in bucket), []))
        result_indices = all_indices - set(ixs)
        params = []
        all_indices_list = list(all_indices)
        to_small_int = lambda x: all_indices_list.index(x)
        for tensor in bucket:
            params.append(tensor.data)
            params.append(list(map(to_small_int, tensor.indices)))
        expr = ','.join(
            ''.join(CHARS[to_small_int(i)] for i in t.indices)
            for t in bucket) +\
                '->'+''.join(CHARS[to_small_int(i)] for i in result_indices)
        print(expr)
        uniq_ids = set(expr) - {',', '-', '>'}
        sizes_dict = {i:2 for i in uniq_ids}
        views = opt_einsum.helpers.build_views(expr, sizes_dict)

        path, info = opt_einsum.contract_path(expr, *views, optimize='auto')
        #print('info', info)

        #print('Largest memory', 8*info.largest_intermediate)
        expect = len(expr.split('->')[1])
        if 2**expect < info.largest_intermediate:
            print(f'WARN: Optimizer auto did not do a good job: '
                  f'expected {2**expect} got {info.largest_intermediate} for contraction')
        #return opt_einsum.contract(contraction, *tensors, optimize=path)


        #print('result_indices',len(result_indices),  result_indices)
        params.append(list(map(to_small_int, result_indices)))
        #print('einsumparams', params)
        #result_data = np.einsum(*params)
        result_data = opt_einsum.contract(*params)

        if len(result_indices) > 0:
            first_index, *_ = result_indices
            tag = first_index.identity
        else:
            tag = 'f'

        result = opt.Tensor(f'E{tag}', result_indices,
                            data=result_data)
        return result

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return np_framework.get_sliced_np_buckets(buckets, data_dict, slice_dict)

    def get_result_data(self, result):
        return result.data

