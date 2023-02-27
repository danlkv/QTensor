import numpy as np
from qtree.optimizer import Tensor

def permute_np_tensor_data(data:np.ndarray, indices_in, indices_out):
    """
    Permute the data of a numpy tensor to the given indices_out.
    
    Returns:
        permuted data
    """
    # permute indices
    out_locs = {idx: i for i, idx in enumerate(indices_out)}
    perm = [out_locs[i] for i in indices_in]
    # permute tensor
    return np.transpose(data, perm)

def get_slice_bounds(slice_dict, indices):
    """Slice a numpy tensor data


    Returns:
        tuple of slice bounds
    """
    slice_bounds = tuple([
        slice_dict.get(i, slice(None)) for i in indices
    ])
    return slice_bounds

def slice_numpy_tensor(data:np.ndarray, indices_in, indices_out, slice_dict):
    """
    Args:
        data : np.ndarray
        indices_in: list of `qtree.optimizer.Var`
        indices_out: list of `qtree.optimizer.Var`
        slice_dict: dict of `qtree.optimizer.Var` to `slice`

    Returns:
        new data, new indices
    """
    slice_bounds = get_slice_bounds(slice_dict, indices_in)
    s_data = data[slice_bounds]
    indices_sliced = [
        i for sl, i in zip(slice_bounds, indices_in) if not isinstance(sl, int)
    ]
    indices_sized = [v.copy(size=size) for v, size in zip(indices_sliced, s_data.shape)]
    assert len(indices_out) == len(s_data.shape)
    assert len(indices_sliced) == len(s_data.shape)
    st_data = permute_np_tensor_data(s_data, indices_sliced, indices_out)
    return st_data, indices_out
