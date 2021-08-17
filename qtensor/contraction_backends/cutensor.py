import qtree
from qtensor.tools.lazy_import import cupy as cp
from qtensor.contraction_backends import ContractionBackend
from qtensor.contraction_backends import CuPyBackend
from cupy import cutensor as cupy_cutensor

mempool = mempool = cp.get_default_memory_pool()

class CuTensorBackend(CuPyBackend):
    
    # Replace all torch methods with cupy's analog
    
    def process_bucket(self, bucket, no_sum=False):
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        for tensor in bucket[1:]:

            expr = qtree.utils.get_einsum_expr(
                list(map(int, result_indices)), list(map(int, tensor.indices))
            )

            a, b, c, mode_a, mode_b, mode_c, desc_a, desc_b, desc_c = self.get_ready(expr, result_data, tensor.data)
            result_data = cupy_cutensor.contraction(1.0, a, desc_a, mode_a, 
                        b, desc_b, mode_b, 0, 
                        c, desc_c, mode_c)

            # Merge and sort indices and shapes
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int)
            )

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
            result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                                data=result_data)
        else:
            result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                                data=cp.sum(result_data, axis=0))
        return result


    @staticmethod
    def get_ready(contraction, a, b):
        # get ready
        inp, out = contraction.split('->')
        size = inp.split(',')
        out_size =cp.full(len(out), 2).tolist() # fill_number = 2

        # TO DO
        mode_a = tuple(size[0])
        mode_b = tuple(size[1])
        mode_c = tuple(out)

        # generate tensor c
        shape_a, shape_b = a.shape, b.shape
        shape_c = tuple(out_size)
        
        c = cp.random.rand(*shape_c).astype(a.dtype)

        # manually cast b to a's type
        b = b.astype(a.dtype)

        # generate tensor descriptor
        desc_a = cupy_cutensor.create_tensor_descriptor(a)
        desc_b = cupy_cutensor.create_tensor_descriptor(b)
        desc_c = cupy_cutensor.create_tensor_descriptor(c)

        
        if not a.flags['C_CONTIGUOUS']:
            a = cp.ascontiguousarray(a)
        if not b.flags['C_CONTIGUOUS']:
            b = cp.ascontiguousarray(b)

        return a, b, c, mode_a, mode_b, mode_c, desc_a, desc_b, desc_c