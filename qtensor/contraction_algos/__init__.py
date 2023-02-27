import qtree
from qtensor.contraction_backends import ContractionBackend

from .merged_bucket_elimination import bucket_elimination as merged_bucket_elimination
from .transposed_bucket_elimination import bucket_elimination as transposed_bucket_elimination
from .transposed_bucket_elimination import test_reverse_order_backend

def bucket_elimination(buckets, backend:ContractionBackend,
                       n_var_nosum=0):
    """
    Algorithm to evaluate a contraction of a large number of tensors.
    """
    if test_reverse_order_backend(backend):
        return transposed_bucket_elimination(buckets, backend.process_bucket, n_var_nosum)
    else:
        return qtree.optimizer.bucket_elimination(buckets, backend.process_bucket, n_var_nosum)


