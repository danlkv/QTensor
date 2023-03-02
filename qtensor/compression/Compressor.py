import io
import sys
import numpy as np
from pathlib import Path
print(Path(__file__).parent/'szx/src/')
sys.path.append(str(Path(__file__).parent/'szx/src/'))
sys.path.append('./szx/src')

try:
    from cuszx_wrapper import cuszx_host_compress, cuszx_host_decompress, cuszx_device_compress, cuszx_device_decompress
except:
    # Silently fail on missing build of cuszx
    pass

CUSZX_BLOCKSIZE = 256

class Compressor():
    def compress(self, data):
        raise NotImplementedError

    def decompress(self, ptr):
        raise NotImplementedError

class NumpyCompressor(Compressor):
    def compress(self, data):
        comp = io.BytesIO()
        np.savez_compressed(comp, data)
        return comp

    def decompress(self, ptr):
        ptr.seek(0)
        return  np.load(ptr)['arr_0']

class CUSZCompressor(Compressor):
    def __init__(self, r2r_error=1e-3, r2r_threshold=1e-3):
        self.r2r_error = r2r_error
        self.r2r_threshold = r2r_threshold

    def compress(self, data):
        import cupy
        if isinstance(data, cupy.ndarray):
            isCuPy = True
        else:
            isCuPy = False
        num_elements = data.size
        # Adapt numele depending on itemsize
        itemsize = data.dtype.itemsize
        num_elements_eff = int(num_elements*itemsize/4) 

        dtype = data.dtype
        cmp_bytes, outSize_ptr = self.cuszx_compress(isCuPy, data, num_elements_eff, self.r2r_error, self.r2r_threshold)
        return (cmp_bytes, num_elements_eff, isCuPy, data.shape, dtype)

    def decompress(self, obj):
        import cupy
        import ctypes
        cmp_bytes, num_elements_eff, isCuPy, shape, dtype = obj
        decompressed_ptr = self.cuszx_decompress(isCuPy, cmp_bytes, num_elements_eff)
        # -- Workaround to convert GPU pointer to int
        p_decompressed_ptr = ctypes.addressof(decompressed_ptr)
        # cast to int64 pointer
        # (effectively converting pointer to pointer to addr to pointer to int64)
        p_decompressed_int= ctypes.cast(p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64))
        decompressed_int = p_decompressed_int.contents
        # --
        mem = cupy.cuda.UnownedMemory(decompressed_int.value, num_elements_eff, self, device_id=0)
        mem_ptr = cupy.cuda.memory.MemoryPointer(mem, 0)
        arr = cupy.ndarray(shape, dtype=dtype, memptr=mem_ptr)
        return arr
    
    ### Compression API with cuSZx ###
    # Parameters:
    # - isCuPy = boolean, true if data is CuPy array, otherwise is numpy array
    # - data = Numpy or Cupy ndarray, assumed to be 1-D, np.float32 type
    # - num_elements = Number of floating point elements in data
    # - r2r_error = relative-to-value-range error bound for lossy compression
    # - r2r_threshold = relative-to-value-range threshold to floor values to zero
    # Returns:
    # - cmp_bytes = Unsigned char pointer to compressed bytes
    # - outSize_ptr = Pointer to size_t representing length in bytes of cmp_bytes
    def cuszx_compress(self, isCuPy, data, num_elements, r2r_error, r2r_threshold):
        
        if not isCuPy:
            cmp_bytes, outSize_ptr = cuszx_host_compress(data, r2r_error, num_elements, CUSZX_BLOCKSIZE, r2r_threshold)
        else:
            cmp_bytes, outSize_ptr = cuszx_device_compress(data, r2r_error, num_elements, CUSZX_BLOCKSIZE, r2r_threshold)
        return cmp_bytes, outSize_ptr

    ### Decompression API with cuSZx ###
    # Parameters:
    # - isCuPy = boolean, true if data is CuPy array, otherwise is numpy array
    # - cmp_bytes = Unsigned char pointer to compressed bytes
    # - num_elements = Number of floating point elements in original data
    # Returns:
    # - decompressed_data = Float32 pointer to decompressed data
    #
    # Notes: Use ctypes to cast decompressed data to Numpy or CuPy type

    def cuszx_decompress(self, isCuPy, cmp_bytes, num_elements):
        if not isCuPy:
            decompressed_data = cuszx_host_decompress(num_elements, cmp_bytes)
        else:
            decompressed_data = cuszx_device_decompress(num_elements, cmp_bytes)

        return decompressed_data
