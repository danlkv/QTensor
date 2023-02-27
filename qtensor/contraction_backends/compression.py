from qtensor.contraction_backends import ContractionBackend
from qtensor.compression import Compressor

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

    def process_bucket(self, bucket):

    def get_sliced_buckets(self, buckets, slice_idx):
        return buckets

    def get_result_data(self, result):
        return result.data
