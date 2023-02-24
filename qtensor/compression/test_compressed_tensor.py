from qtensor.compression import CompressedTensor
from qtensor.compression.CompressedTensor import Compressor, CUSZCompressor
from qtree.optimizer import Var
from qtree.system_defs import NP_ARRAY_TYPE
import pytest
import numpy as np

def test_empty_tensor():
    shape = (2, 3, 4)
    indices = [Var(i, size=s) for i, s in enumerate(shape)]
    t = CompressedTensor.empty("myT", indices)
    assert t.name == "myT"
    assert t.indices == tuple(indices)
    assert t.shape == shape
    assert t.data is not None
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
    assert S.data is not None
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
    assert S.data is not None
    assert np.allclose(t.get_chunk([1, 2]), S.data)

@pytest.mark.parametrize(argnames=["shape", "compressor"],
                         argvalues=[
                             ((2, 3, 4), Compressor()),
                             ((2, 3, 4), CUSZCompressor()),
                             ((2,)*20, CUSZCompressor())
                        ]
                        )
def test_compressors(shape, compressor):
    import cupy
    indices = [Var(i, size=s) for i, s in enumerate(shape)]
    data = cupy.random.randn(*shape)
    print("Data size", data.nbytes)
    t = CompressedTensor("myT", indices, data=data, compressor=compressor)
    t.compress_indices([indices[0]])

    s = t[1]
    print('got chunk')
    assert s.data is not None
    assert np.allclose(t.get_chunk([1]), s.data)

