import itertools
import numpy as np
import io
from qtree.optimizer import Tensor
from qtree.system_defs import NP_ARRAY_TYPE
import sys
sys.path.append("./szx/src")

from cuszx_wrapper import cuszx_host_compress, cuszx_host_decompress, cuszx_device_compress, cuszx_device_decompress

CUSZX_BLOCKSIZE = 256

def iterate_indices(indices: list):
    if len(indices)==0:
        return [tuple()]
    ranges = [range(v.size) for v in indices]
    return itertools.product(*ranges)

class Compressor():
    def compress(self, data):
        print(f"Compressing len {data.size}")
        comp = io.BytesIO()
        np.savez_compressed(comp, data)
        return comp

    def decompress(self, ptr):
        ptr.seek(0)
        print(f"Loading arr.")
        return  np.load(ptr)['arr_0']

class CUSZCompressor():
    def compress(self, data):
        import cupy
        if isinstance(data, cupy.ndarray):
            isCuPy = True
        else:
            isCuPy = False
        num_elements = data.size
        r2r_error = 0.01
        r2r_threshold = 0.01
        cmp_bytes, outSize_ptr = self.cuszx_compress(isCuPy, data.flatten(), num_elements, r2r_error, r2r_threshold)
        print("returning compressed data")
        return (cmp_bytes, num_elements, isCuPy, data.shape)

    def decompress(self, obj):
        import cupy
        import ctypes
        cmp_bytes, num_elements, isCuPy, shape = obj
        decompressed_ptr = self.cuszx_decompress(isCuPy, cmp_bytes, num_elements)
        # -- Workaround to convert GPU pointer to int
        p_decompressed_ptr = ctypes.addressof(decompressed_ptr)
        # cast to int64 pointer
        # (effectively converting pointer to pointer to addr to pointer to int64)
        p_decompressed_int= ctypes.cast(p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64))
        decompressed_int = p_decompressed_int.contents
        # --
        mem = cupy.cuda.UnownedMemory(decompressed_int.value, num_elements*8, self, device_id=0)
        mem_ptr = cupy.cuda.memory.MemoryPointer(mem, 0)
        arr = cupy.ndarray(shape, dtype=np.float64, memptr=mem_ptr)
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

class CompressedTensor(Tensor):
    """
    Extension of the Tensor class that holds compressed data

    The data array is split along several indices S into 2^|S| parts

    """
    def __init__(self, name, indices,
                 data_key=None, data=None,
                 slice_indices=[],
                 compressor=Compressor()
                ):
        """
        Initialize the tensor
        name: str,
              the name of the tensor. Used only for display/convenience.
              May be not unique.
        indices: tuple,
              Indices of the tensor
        shape: tuple,
              shape of a tensor
        data_key: int
              Key to find tensor's data in the global storage
        data: np.array
              Actual data of the tensor. Default None.
              Usually is not supplied at initialization.
        slice_indices: list[Var]
            indices along which the tensor is split into chunks
        """
        super().__init__(name, indices, data_key=data_key, data=data)
        self.slice_indices = slice_indices
        self.compressor = compressor
        if data is not None:
            self._dtype = data.dtype
        else:
            self._dtype = None

    @classmethod
    def empty(cls, name, indices, slice_indices=[], compressor=Compressor(), dtype:type=NP_ARRAY_TYPE):
        t = super().empty(name, indices, dtype)
        t.compressor = compressor
        if slice_indices:
            t.compress_indices(slice_indices)
        return t

    def compress_indices(self, indices: list):
        """
        Slice the self.data along dimensions in `indices`,
        store them compressed

        Does not support compressing when already compressed
        """
        slice_dict = {
            i: slice(None) for i in self.indices
        }
        data_chunks = []
        for ivals in iterate_indices(indices):
            for ix, ival in zip(indices, ivals):
                slice_dict[ix] = ival# slice(ival, ival+1)
            dslice = self.data[tuple(slice_dict[i] for i in self.indices)]

            data_chunks.append(
                self.compressor.compress(dslice)
            )
            del dslice
        self._data = data_chunks
        self.slice_indices = indices

    @property
    def dtype(self):
        """
        DataType of wrapped chunks.
        """
        return self._dtype

    @property
    def array_indices(self):
        return [x for x in self.indices if x not in self.slice_indices]

    def get_chunk(self, ivals):
        dims = [v.size for v in self.slice_indices]
        if len(ivals)==0:
            flat_ix = 0
        else:
            flat_ix = np.ravel_multi_index(ivals, dims)
        ptr = self._data[flat_ix]
        return self.compressor.decompress(ptr)

    def set_chunk(self, ivals, chunk: np.ndarray):
        # -- Check for consistent data types between chunks
        if self._dtype is None:
            self._dtype = chunk.dtype
        else:
            assert self.dtype == chunk.dtype, f"Chunk dtype {chunk.dtype} does not match tensor dtype {self.dtype}"
        # --

        if self._data is None:
            self._data = np.empty(2**len(self.slice_indices), dtype=object)
        dims = [v.size for v in self.slice_indices]
        if len(ivals)==0:
            flat_ix = 0
        else:
            flat_ix = np.ravel_multi_index(ivals, dims)
        self._data[flat_ix] = self.compressor.compress(chunk)

    def __getitem__(self, key):
        """
        Get a slice of the tensor along the indices in `key`
        Currently slicing over all compressed indices is required.
        Slices over compressed indices must be ints
        """
        slices_ints, new_indices = self._parse_getitem_key(key)
        slice_dict = {}
        chunk_slices_ints = []
        compression_ints = []
        for ix, ival in zip(self.indices, slices_ints):
            slice_dict[ix] = ival
            if ix in self.slice_indices:
                compression_ints.append(ival)
            else:
                chunk_slices_ints.append(ival)
        chunk = self.get_chunk(compression_ints)
        new_name = f"{self.name}[sliced]"
        # careful: chunk will not be collected even if slice is small
        chunk_slice = chunk[tuple(chunk_slices_ints)]
        return Tensor(new_name, new_indices, data=chunk_slice)


    def __str__(self):
        array_ix = ','.join(map(str, self.array_indices))
        split_ix= ','.join(map(str, self.slice_indices))
        return f'{self._name}{{{split_ix}}}({array_ix})'

    def copy(self, name=None, indices=None, data_key=None, data=None):
        raise NotImplementedError()

    def __repr__(self):
        return self.__str__()



