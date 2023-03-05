from qtensor.contraction_backends.common import slice_numpy_tensor
import numpy as np
from qtree.optimizer import Var

def test_slice_numpy_tensor():
    shape = (2, 3, 4, 5)
    indices_in = [Var(i, size=s) for i, s in enumerate(shape)]
    data = np.random.rand(*shape)
    data_ref = data.copy()
    slice_dict = {
        indices_in[0]: slice(None),
        indices_in[1]: slice(1, 3),
        indices_in[2]: 1,
        indices_in[3]: slice(3, 4),
    }
    indices_out = [indices_in[3], indices_in[1], indices_in[0]]
    new_data, new_indices = slice_numpy_tensor(
        data, indices_in, indices_out, slice_dict
    )
    assert new_data.shape == (1, 2, 2)
    assert new_indices == indices_out
    assert np.allclose(data,  data_ref)
    assert not np.allclose(new_data , data_ref[:, 1:3, 1, 3:4])
    assert np.allclose(new_data , data_ref[:, 1:3, 1, 3:4].transpose(2, 1, 0))
    assert np.allclose(new_data , data_ref.transpose()[3:4, 1, 1:3, :])

def test_slice_torch_tensor():
    import torch
    shape = (2, 3, 4, 5)
    indices_in = [Var(i, size=s) for i, s in enumerate(shape)]
    data = torch.randn(*shape)
    data_ref = data.clone()
    slice_dict = {
        indices_in[0]: slice(None),
        indices_in[1]: slice(1, 3),
        indices_in[2]: 1,
        indices_in[3]: slice(3, 4),
    }
    indices_out = [indices_in[3], indices_in[1], indices_in[0]]
    new_data, new_indices = slice_numpy_tensor(
        data, indices_in, indices_out, slice_dict
    )
    assert isinstance(new_data, torch.Tensor)
    assert new_data.shape == (1, 2, 2)
    assert new_indices == indices_out
    assert np.allclose(data,  data_ref)
    assert not np.allclose(new_data , data_ref[:, 1:3, 1, 3:4])
    assert np.allclose(new_data , data_ref[:, 1:3, 1, 3:4].permute(2, 1, 0))
