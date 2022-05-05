from qtensor.contraction_backends import ContractionBackend
"""
class MixedBe(ConBE):
    be1: cpu_be
    be2: gpu_be

    def get_sliced_bucket():
        normal slicing
            either use be1_get_sliuced or naive implementation
            np.array for convinience
    
    def process_bucket():
        - Check input bucket width
        - If larger than 8, use gpu_be.process(bucket)
        - Else, use less than 8, use cpu_be.process(bucket)

    def get_result_data():
        - always use gpu_be's get_result
        - so that the gpu_be shall handle the gpu-cpu transfer all the time

"""


'''
Input:  Array of either (indices) or a tensor
        If a tensor, use t.indices for conversion
'''
def bucketWidth(bucket):
    bucket_width = 0
    for tensor in bucket:
        tensor_len = 0
        if type(tensor) is tuple:
            tensor_len = len(tensor)
            if tensor_len > bucket_width:
                bucket_width = tensor_len
        else:
            tensor_len = len(tensor.indices)
            if tensor_len > bucket_width:
                bucket_width = tensor_len
    return bucket_width


def AccurateBucketWidth(bucket):
    indices_set = set()
    for tensor in bucket:
        if type(tensor) is tuple:
            for index in tensor:
                indices_set.add(index)
        else:
            for index in tensor.indices:
                indices_set.add(index)

    return len(indices_set)
    



'''
I/O: Actual BE Objects -> Wrapped Class
TODO: process_bucket_merged() for its own, when when we test the merged simulators, we create the merged simulator
'''

class MixBackend(ContractionBackend):

    def __init__(self, cpu_be, gpu_be, watershed):
        self.cpu_be = cpu_be
        self.gpu_be = gpu_be
        self.watershed = watershed
    
    def process_bucket(self, bucket, no_sum = False):
        # bucket_width = bucketWidth(bucket)
        accc_width = AccurateBucketWidth(bucket)
        # if accc_width != bucket_width:
        #      print(accc_width, bucket_width)
        if accc_width >= self.watershed:
            #print(f"In GPU, width {accc_width}")
            return self.gpu_be.process_bucket(bucket, no_sum)
        else:
            return self.cpu_be.process_bucket(bucket, no_sum)

    def process_bucket_merged(self,ixs, bucket, no_sum=False):
        accu_width = AccurateBucketWidth(bucket)
        if accu_width >= self.watershed:
            return self.gpu_be.process_bucket_merged(ixs, bucket, no_sum)
        else:
            return self.cpu_be.process_bucket_merged(ixs, bucket, no_sum)

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return self.cpu_be.get_sliced_buckets(buckets, data_dict, slice_dict)
    
    def get_result_data(self, result):
        return self.gpu_be.get_result_data(result)
