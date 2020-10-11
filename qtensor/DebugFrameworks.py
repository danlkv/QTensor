""" Farmeworks that print a lot of info.
This file is meant to be temporary and will not be updated
"""
import sys
import time
from functools import reduce
import numpy as np
from qtree import np_framework
from qtree import optimizer as opt
from qtensor.ProcessingFrameworks import BucketBackend, PerfBackend

class MockModule:
    def __getattribute__(self, attr):
        # Fail spectacularly
        raise ImportError(f'Module tcontract is not imported! Please install it and try again.')

tcontract = MockModule()
try:
    import tcontract
except ImportError:
    pass

class _CMKLExtendedBackend(BucketBackend):
    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return np_framework.get_sliced_np_buckets(buckets, data_dict, slice_dict)

    def process_bucket(self, bucket, no_sum=False):
        result_indices = bucket[0].indices
        result_data = bucket[0].data

        # -- Contract first n-1 bucketns
        def merge_with_result(result_data, result_indices, tensor):
            # ---- Prepare inputs: transpose + reshape
            ixa, ixb = result_indices, tensor.indices
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

            c = np.empty((n, m, k), dtype=np.complex128)
            start = time.time()
            print(f'Starting debug_mkl_contract, input sizes: {a.size} {b.size} output: {c.size}', file=sys.stderr)
            tcontract.debug_mkl_contract_complex(a, b, c)
            end  = time.time()
            print(f'After debug_mkl_contract, duration: {end - start}', file=sys.stderr)

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

        for tensor in bucket[1:-1]:
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
        last_tensor = bucket[-1]

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
        c = np.empty((F, M, N), dtype=np.complex128)
        start = time.time()
        print(f'Starting debug_mkl_contract_sum, input sizes: {a.size} {b.size} output: {c.size}', file=sys.stderr)
        tcontract.debug_mkl_contract_sum(a, b, c)
        end  = time.time()
        print(f'After debug_mkl_contract_sum, duration: {end - start}', file=sys.stderr)

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

    def get_result_data(self, result):
        return result.data

class DebugMKLBackend(PerfBackend):
    Backend = _CMKLExtendedBackend
    # Just use print by default
    def __init__(self, *args, print=True, num_lines=20, **kwargs):
        super().__init__(*args, print=print, num_lines=num_lines, **kwargs)
