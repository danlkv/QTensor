import itertools
import numpy as np
import qtree
import io
from qtree.optimizer import Tensor, Var

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

    def set_chunk(self, ivals, chunk:np.array):
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



