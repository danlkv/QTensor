from qtensor.contraction_backends import ContractionBackend
from functools import reduce
import qtree
import qtree.optimizer as opt
import qtensor
from qtree import np_framework
import opt_einsum
import numpy as np
import string

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

class TransposedBackend(ContractionBackend):
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

    def pairwise_sum_contract(self, ixa, a, ixb, b, ixout):
        #ixa, ixb, out = self._to_letters(ixa, ixb, ixout)
        out = ixout
        ixs = ixa, ixb
        tensors = a, b

        # \sum_k A_{kfm} * B_{kfn} = C_{fmn}
        common = set(ixs[0]).intersection(set(ixs[1]))
        kix = common - set(out)
        fix = common - kix
        # -- 1-tensor summation

        summed_ixs = (set(list(ixs[0]) + list(ixs[1]))) - set(out)
        summed_ixs =  summed_ixs - set(kix)
        if summed_ixs.isdisjoint(set(ixs[0])):
            t = tensors[1].sum(axis=tuple(list(ixs[1]).index(i) for i in summed_ixs))
            i = set(ixs[1]) - summed_ixs
            tensors = tensors[0], t
            ixs = ixs[0], i
        else:
            t = tensors[0].sum(axis=tuple(list(ixs[0]).index(i) for i in summed_ixs))
            i = set(ixs[0]) - summed_ixs
            ixs = i, ixs[1]
            tensors = t, tensors[1]
        # --

        mix = set(ixs[0]) - common
        nix = set(ixs[1]) - common
        common = list(kix) + list(fix)
        a = tensors[0].transpose(*[
            list(ixs[0]).index(x) for x in common + list(mix)
        ])

        b = tensors[1].transpose(*[
            list(ixs[1]).index(x) for x in common + list(nix)
        ])

        k, f, m, n = [self._get_index_space_size(*ix)
                      for ix in (kix, fix, mix, nix)
                     ]
        #print('kfmn', k, f, m, n)
        #print('ixa', ixa, 'ixb', ixb)
        a = a.reshape(k, f, m)
        b = b.reshape(k, f, n)
        G = np.einsum('ijk,ijl->jkl', a, b)
        if len(out):
            #print('out ix', out, 'kfmnix', kix, fix, mix, nix)
            G = G.reshape(*self._get_index_sizes(*out))
        current_ord_ = list(fix) + list(mix) + list(nix)
        G = G.transpose(*[current_ord_.index(i) for i in out])

        return G


    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        #bucket = list(reversed(bucket))
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        all_indices = set(sum((list(t.indices) for t in bucket), []))
        result_indices = all_indices - set(ixs)

        cum_data, cum_indices = bucket[0].data, bucket[0].indices
        #print('bucket', [len(t.indices) for t in bucket])
        #print('subsets', [set(t.indices).issubset(all_indices) for t in bucket])
        #print('contracted', len(ixs), ixs)
        for tensor in bucket[1:-1]:
            next_indices = set(cum_indices).union(tensor.indices)
            #print('next ix size', len(next_indices), len(cum_indices), len(tensor.indices))
            cum_data = self.pairwise_sum_contract(
                cum_indices, cum_data, tensor.indices, tensor.data, next_indices
            )
            cum_indices = next_indices

        tensor = bucket[-1]
        result_data = self.pairwise_sum_contract(
            cum_indices, cum_data, tensor.indices, tensor.data, result_indices
        )

        tag = str(list(ixs)[0])

        result = opt.Tensor(f'E{tag}', result_indices,
                            data=result_data)
        return result

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return np_framework.get_sliced_np_buckets(buckets, data_dict, slice_dict)

    def get_result_data(self, result):
        return result.data

