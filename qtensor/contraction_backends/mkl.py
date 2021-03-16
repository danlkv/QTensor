import numpy as np
from functools import reduce
from qtree import optimizer as opt
import qtensor
from qtensor.contraction_backends import ContractionBackend
from qtree import np_framework
from qtensor.tools.lazy_import import tcontract
import pyrofiler

import string

CHARS = string.ascii_lowercase + string.ascii_uppercase

class CMKLExtendedBackend(ContractionBackend):
    times = []
    times_all = []
    times_post = []
    times_pre = []
    times_sre = []
    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return np_framework.get_sliced_np_buckets(buckets, data_dict, slice_dict)

    def process_bucket(self, bucket, no_sum=False):

        # -- Contract first n-1 bucketns
        def merge_with_result(result_data, result_indices, tensor):
            # ---- Prepare inputs: transpose + reshape
            with pyrofiler.timing(callback=lambda x: self.times_pre.append(x)):
                ixa, ixb = result_indices, tensor.indices
                with pyrofiler.timing(callback=lambda x: self.times_sre.append(x)):
                    common_ids = sorted(list(set.intersection(set(ixa), set(ixb))), key=int)
                distinct_a = [x for x in ixa if x not in common_ids]
                distinct_b = [x for x in ixb if x not in common_ids]
                transp_a = [ixa.index(x) for x in common_ids+distinct_a]
                transp_b = [ixb.index(x) for x in common_ids+distinct_b]
                a = result_data.transpose(transp_a)
                b = tensor.data.transpose(transp_b)
                n, m, k = 2**len(common_ids), 2**len(distinct_a), 2**len(distinct_b)
                a = a.reshape(n, m)
                b = b.reshape(n, k)
            # ----

            with pyrofiler.timing(callback=lambda x: self.times_all.append(x)):
                c = np.empty((n, m, k), dtype=np.complex128)
            with pyrofiler.timing(callback=lambda x: self.times.append(x)):
                tcontract.mkl_contract_complex(a, b, c)

            # ---- Post-process output
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int)
            )
            ixc = common_ids + distinct_a + distinct_b
            assert len(result_indices) == len(ixc), 'Wrong transposition, please submit an issue'
            transp_c = [ixc.index(x) for x in result_indices]
            result_data = c.reshape(*[2 for _ in result_indices])
            result_data = result_data.transpose(transp_c)
            return result_data, result_indices
            # ----

        result_indices = bucket[-1].indices
        result_data = bucket[-1].data

        for tensor in reversed(bucket[1:-1]):
            result_data, result_indices = merge_with_result(result_data, result_indices, tensor)
        # --


        if len(result_indices) > 0:
            tag = result_indices[0].identity
        else:
            tag = 'f'

        if no_sum:
            if len(bucket)>1:
                last_tensor = bucket[-1]
                result_data, result_indices = merge_with_result(result_data, result_indices, last_tensor)

            result = opt.Tensor(f'E{tag}', result_indices,
                                data=result_data)
            return result

        if len(bucket)<2:
            result = opt.Tensor(f'E{tag}', result_indices[1:],
                                data=np.sum(result_data, axis=0))
            return result
        last_tensor = bucket[0]

        # -- Contract with summation
        ixa, ixb = result_indices, last_tensor.indices
        # ---- Prepare inputs: transpose + reshape
        k, fm = result_indices[:1], result_indices[1:]
        fn = last_tensor.indices[1:]

        f = tuple(sorted(list(set.intersection(set(fm), set(fn))), key=int))
        # Sets don't store order, so use lists. Do we need order here?
        m = tuple([x for x in fm if x not in f])
        n = tuple([x for x in fn if x not in f])
        transp_a = [ixa.index(x) for x in k+f+m]
        transp_b = [ixb.index(x) for x in k+f+n]
        a = result_data.transpose(transp_a)
        b = last_tensor.data.transpose(transp_b)
        shapes_a = {i:s for i,s in zip(k+f+m, a.shape)}
        shapes_b = {i:s for i,s in zip(k+f+n, b.shape)}
        shapes = {**shapes_b, **shapes_a}
        K, F, M, N = [reduce(np.multiply, (shapes[i] for i in x), 1) for x in (k, f, m, n)]
        a = a.reshape(K, F, M)
        b = b.reshape(K, F, N)
        # ----

        # \sum_k A_{kfm} * B_{kfn} = C_{fmn}
        with pyrofiler.timing(callback=lambda x: self.times_all.append(x)):
            c = np.empty((F, M, N), dtype=np.complex128)
        with pyrofiler.timing(callback=lambda x: self.times.append(x)):
            tcontract.mkl_contract_sum(a, b, c)

        with pyrofiler.timing(callback=lambda x: self.times_post.append(x)):
            # ---- Post-process output
            result_indices = tuple(sorted(
                set(result_indices + last_tensor.indices),
                key=int)
            )
            assert result_indices[0] == k[0], 'Broken ordering, please report'
            result_indices = result_indices[1:]
            ixc = f + m + n
            assert len(result_indices) == len(ixc), 'Wrong transposition, please submit an issue'
            result_data = c.reshape([shapes[i] for i in ixc])
            transp_c = [ixc.index(x) for x in result_indices]
            result_data = result_data.transpose(transp_c)
            # ----
            # --
            result = opt.Tensor(f'E{tag}', result_indices, data=result_data)
        return result

    def _expr_to_ixs(self, expr):
        if isinstance(expr, str):
            inp, out = expr.split('->')
        else:
            pass
        ixs = inp.split(',')
        return ixs, out
    def _to_letters(self, *ixs):
        all_indices = list(qtensor.utils.merge_sets(set(x) for x in ixs))
        mapping = {i:s for i,s in zip(all_indices, CHARS)}
        convert = lambda x: [mapping[i] for i in x]
        return list(map(convert, ixs))

    def _get_index_sizes(self, *ixs):
        try:
            sizes = [ i.size for i in ixs ]
        except AttributeError:
            sizes = [2] * len(ixs)
        return sizes

    def _get_index_space_size(self, *ixs):
        sizes = self._get_index_sizes(*ixs)
        return reduce(np.multiply, sizes, 1)

    #@profile
    def pairwise_sum_contract(self, ixa, a, ixb, b, ixout):
        out = ixout
        common = set(ixa).intersection(set(ixb))
        # -- sum indices that are in one tensor only
        all_ix = set(ixa+ixb)
        sum_ix = all_ix - set(out)
        a_sum = sum_ix.intersection(set(ixa) - common)
        b_sum = sum_ix.intersection(set(ixb) - common)
        #print('ab', ixa, ixb)
        #print('all sum', sum_ix, 'a/b_sum', a_sum, b_sum)
        if len(a_sum):
            a = a.sum(axis=tuple(ixa.index(x) for x in a_sum))
            ixa = [x for x in ixa if x not in a_sum]
        if len(b_sum):
            b = b.sum(axis=tuple(ixb.index(x) for x in b_sum))
            ixb = [x for x in ixb if x not in b_sum]
        tensors = a, b
        # --

        ixs = ixa, ixb
        common = set(ixs[0]).intersection(set(ixs[1]))

        # \sum_k A_{kfm} * B_{kfn} = C_{fmn}
        mix = set(ixs[0]) - common
        nix = set(ixs[1]) - common
        kix = common - set(out)
        fix = common - kix
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
        with pyrofiler.timing(callback=lambda x: self.times_pre.append(x)):
            a = a.reshape(k, f, m)
            b = b.reshape(k, f, n)
        with pyrofiler.timing(callback=lambda x: self.times_all.append(x)):
            c = np.empty((f, m, n), dtype=np.complex128)
        with pyrofiler.timing(callback=lambda x: self.times.append(x)):
            tcontract.mkl_contract_sum(a,b,c)
        if len(out):
            #print('out ix', out, 'kfmnix', kix, fix, mix, nix)
            c = c.reshape(*self._get_index_sizes(*out))
        #print('outix', out, 'res', c.shape, 'kfmn',kix, fix, mix, nix)

        current_ord_ = list(fix) + list(mix) + list(nix)
        c = c.transpose(*[current_ord_.index(i) for i in out])
        return c


    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        all_indices = set(sum((list(t.indices) for t in bucket), []))
        result_indices = all_indices - set(ixs)

        cum_data, cum_indices = bucket[0].data, bucket[0].indices
        for i, tensor in enumerate(bucket[1:-1]):
            next_indices = set(cum_indices).union(tensor.indices)

            #-- contract indices specific to this pair, if any
            # note that it may be better to reorder pairs of tensors
            # sorting by largest set of pair-specific indices
            _seta, _setb = set(cum_indices), set(tensor.indices)
            _intersec = _seta.intersection(_setb)
            _others = set(sum((list(t.indices) for t in bucket[i+2:]),[]))
            specific_indices = _intersec - _others
            #print('specific indices', specific_indices, 'intersection', _intersec)

            # allow to contract only those indices that are not requested in resutlt
            specific_indices = specific_indices.intersection(set(ixs))

            if len(specific_indices):
                print('[D] Found specific_indices', specific_indices, 'ixs', ixs)
            #--
            next_indices = next_indices - specific_indices
            cum_data = self.pairwise_sum_contract(
                cum_indices, cum_data, tensor.indices, tensor.data, next_indices
            )
            cum_indices = tuple(next_indices)

        tensor = bucket[-1]
        result_data = self.pairwise_sum_contract(
            cum_indices, cum_data, tensor.indices, tensor.data, result_indices
        )

        tag = str(list(ixs)[0])

        result = opt.Tensor(f'E{tag}', result_indices,
                            data=result_data)
        return result

    def get_result_data(self, result):
        return result.data
