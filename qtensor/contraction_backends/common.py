import numpy as np
import qtree
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
    #print("indices_sized", indices_sized)
    #print("Slice bounds", slice_bounds)
    #print("Slice dict", slice_dict)
    #print("data shape, sliced data shape", data.shape, s_data.shape)
    indices_out = [v for v in indices_out if not isinstance(slice_dict.get(v, None), int)]
    assert len(indices_sliced) == len(s_data.shape)
    st_data = permute_np_tensor_data(s_data, indices_sliced, indices_out)
    return st_data, indices_out

def get_einsum_expr(idx1, idx2, contract=0):
    """
    Takes two tuples of indices and returns an einsum expression
    to evaluate the sum over repeating indices

    Parameters
    ----------
    idx1 : list-like
          indices of the first argument
    idx2 : list-like
          indices of the second argument

    Returns
    -------
    expr : str
          Einsum command to sum over indices repeating in idx1
          and idx2.
    """
    result_indices = sorted(list(set(idx1 + idx2)), reverse=True)
    # remap indices to reduce their order, as einsum does not like
    # large numbers
    idx_to_least_idx = {old_idx: new_idx for new_idx, old_idx
                        in enumerate(result_indices)}
    result_indices = result_indices[:len(result_indices)-contract]

    str1 = ''.join(qtree.utils.num_to_alpha(idx_to_least_idx[ii]) for ii in idx1)
    str2 = ''.join(qtree.utils.num_to_alpha(idx_to_least_idx[ii]) for ii in idx2)
    str3 = ''.join(qtree.utils.num_to_alpha(idx_to_least_idx[ii]) for ii in result_indices)
    return str1 + ',' + str2 + '->' + str3
