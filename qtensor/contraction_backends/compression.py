from qtensor.contraction_backends import ContractionBackend
from qtensor.compression import Compressor
from qtensor.compression.compressed_contraction import compressed_contract
from qtensor.contraction_backends.common import slice_numpy_tensor
from qtree.optimizer import Tensor

class CompressionBackend(ContractionBackend):
    """
    Compression bucket contraction backend.

    This backend "decorates" another backend, by using compression in 
    pairwise contraction. If the result tensor has more than `max_tw` indices,
    it is sliced and the contraction result is compressed before proceeding to
    next slice.
    """
    def __init__(self, backend, compressor:Compressor, max_tw:int):
        """
        Arguments:
            backend: the backend to use for contraction
            compressor: the compressor to use for compression
            max_tw: threshold for triggering compression.

        """
        self.backend = backend
        self.compressor = compressor
        self.max_tw = max_tw

    def process_bucket(self, bucket, no_sum=False):
        """
        Process a bucket.

        This uses `self.backend.process_bucket` in combination with
        compression.compressed_contraction.compressed_contract
        """
        bucket.sort(key=lambda x: len(x.indices))
        accum = bucket[0]
        for t in bucket[1:-1]:
            contract_ixs = set().union(*[t.indices, accum.indices])
            accum = compressed_contract(
                accum, t, contract_ixs, self.max_tw, self.compressor
            )
        if len(bucket)>1:
            t = bucket[-1]
            contract_ixs = sorted(
                set().union(*[t.indices, accum.indices])
                , key=int, reverse=True
            )
            contract_ixs = contract_ixs[:-1]
            accum = compressed_contract(
                accum, t, contract_ixs, self.max_tw, self.compressor
            )
            return accum
        else:
            # This assumes large buckets with one element don't exist
            result_data = accum.data.sum(axis=-1)
            return Tensor(accum.name, accum.indices[:-1], data=result_data)

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        """
        Slice buckets accounding to `slice_dict`

        This delegates to `self.backend`, assuming that buckets don't have 
        tensors with more than `self.max_tw` indices.
        """
        # Note: to support large tensors (more than `max_tw`), 
        # just iterate through sliced bucket tensors and compress if needed
        return self.backend.get_sliced_buckets(buckets, data_dict, slice_dict)

    def get_result_data(self, result):
        """
        Get result data from `result` tensor.

        This assumes that the result has at most `self.max_tw` indices.
        """
        return self.backend.get_result_data(result)
