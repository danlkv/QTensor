from qtensor.compression import CompressedTensor
from qtree.optimizer import Tensor, Var
from qtree.system_defs import NP_ARRAY_TYPE
import numpy as np

def test_empty_tensor():
    shape = (2, 3, 4)
    indices = [Var(i, size=s) for i, s in enumerate(shape)]
    t = CompressedTensor.empty("myT", indices)
    assert t.name == "myT"
    assert t.indices == tuple(indices)
    assert t.shape == shape
    assert t.data.shape == shape
    assert t.data.dtype == NP_ARRAY_TYPE

    t.compress_indices([indices[0]])
    assert t.dtype == NP_ARRAY_TYPE


def test_slice_tensor():
    shape = (2, 3, 4)
    indices = [Var(i, size=s) for i, s in enumerate(shape)]
    t = CompressedTensor.empty("myT", indices, dtype=np.uint32)
    t.compress_indices([indices[0]])
    S = t[{indices[0]: 1, indices[1]: slice(0, 1)}]
    assert S.data.shape == (1, 4)
    assert indices[0] not in S.indices
    assert int(indices[1]) == int(S.indices[0])
    assert indices[1] != S.indices[0]
    assert indices[2] in S.indices
    assert S.indices[1].size == 4
    assert np.allclose(t.get_chunk([1])[0:1], S.data)

    t = CompressedTensor.empty("myT", indices, dtype=np.uint32)
    t.compress_indices([indices[0], indices[1]])
    S = t[1, 2]
    assert indices[1] not in S.indices
    assert np.allclose(t.get_chunk([1, 2]), S.data)

