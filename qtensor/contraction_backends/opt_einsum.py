from qtensor.contraction_backends import ContractionBackend
from functools import reduce
import qtree
import qtree.optimizer as opt
import qtensor
from qtree import np_framework
import numpy as np
import string
from qtensor.tools.lazy_import import opt_einsum

CHARS = string.ascii_lowercase + string.ascii_uppercase

def opt_einsum_info(expr):
    uniq_ids = set(expr) - {',', '-', '>'}
    sizes_dict = {i:2 for i in uniq_ids}
    views = opt_einsum.helpers.build_views(expr, sizes_dict)
    path, info = opt_einsum.contract_path(expr, *views, optimize='auto')
    return path, info

def get_einsum_expr(bucket, all_indices_list, result_indices):
    # converting elements to int will make stuff faster, 
    # but will drop support for char indices
    # all_indices_list = [int(x) for x in all_indices]
    # to_small_int = lambda x: all_indices_list.index(int(x))
    to_small_int = lambda x: all_indices_list.index(x)
    expr = ','.join(
        ''.join(CHARS[to_small_int(i)] for i in t.indices)
        for t in bucket) +\
            '->'+''.join(CHARS[to_small_int(i)] for i in result_indices)
    return expr

class OptEinusmBackend(ContractionBackend):
    def __init__(self):
        super().__init__()
        #self.pbar = tqdm(desc='Buckets', position=2)
        #self.status_bar = tqdm(desc='Current status', position=3, bar_format='{desc}')

    def process_bucket(self, bucket, no_sum=False):
        all_indices = list(qtensor.utils.merge_sets(set(x.indices) for x in bucket))
        all_indices = sorted(all_indices, key=int)
        result_indices = all_indices[1:]
        contract_index = all_indices[0]
        #print('contract_index', contract_index, no_sum)
        res = self.process_bucket_merged([contract_index], bucket, no_sum=no_sum)
        # Convert indices to growing, bucket elimination expects this
        tdata = res.data.transpose(*[res.indices.index(x) for x in result_indices])
        res = qtree.optimizer.Tensor(name=res.name, indices=tuple(result_indices), data=tdata)
        return res

    def _expr_to_ixs(self, expr):
        if isinstance(expr, str):
            inp, out = expr.split('->')
        else:
            pass
        ixs = inp.split(',')
        return ixs, out

    def _get_index_sizes(self, *ixs):
        try:
            sizes = [ i.size for i in ixs ]
        except AttributeError:
            sizes = [2] * len(ixs)
        return sizes

    def _get_index_space_size(self, *ixs):
        sizes = self._get_index_sizes(*ixs)
        return reduce(np.multiply, sizes, 1)

    def _to_letters(self, *ixs):
        all_indices = list(qtensor.utils.merge_sets(set(x) for x in ixs))
        mapping = {i:s for i,s in zip(all_indices, CHARS)}
        convert = lambda x: [mapping[i] for i in x]
        return list(map(convert, ixs))

    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        all_indices = set(sum((list(t.indices) for t in bucket), []))
        result_indices = all_indices - set(ixs)
        expr = get_einsum_expr(bucket, list(all_indices), list(result_indices))
        tensor_data =  [t.data for t in bucket]
        #print(opt_einsum_info(expr))
        result_data = opt_einsum.contract(expr, *tensor_data)
        tag = str(list(ixs)[0])
        result = opt.Tensor(f'E{tag}', result_indices,
                            data=result_data)
        return result

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return np_framework.get_sliced_np_buckets(buckets, data_dict, slice_dict)

    def get_result_data(self, result):
        return result.data

