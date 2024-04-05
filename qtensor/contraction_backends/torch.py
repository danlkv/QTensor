from qtensor.tools.lazy_import import torch
import qtree
import numpy as np
from functools import reduce
from qtree import np_framework
from qtensor.contraction_backends import ContractionBackend
from .common import get_slice_bounds, get_einsum_expr, slice_numpy_tensor
import string
from loguru import logger
CHARS = string.ascii_lowercase + string.ascii_uppercase

def qtree2torch_tensor(tensor, data_dict):
    """ Converts qtree tensor to pytorch tensor using data dict"""
    if isinstance(tensor.data, torch.Tensor):
        return tensor
    if tensor.data is not None:
        data = tensor.data
    else:
        data = data_dict[tensor.data_key]
    torch_t = torch.from_numpy(data)
    data_dict[tensor.data_key] = torch_t
    return tensor.copy(data=torch_t)

def get_einsum_expr_bucket(bucket, all_indices_list, result_indices):
    # converting elements to int will make stuff faster, 
    # but will drop support for char indices
    # all_indices_list = [int(x) for x in all_indices]
    # to_small_int = lambda x: all_indices_list.index(int(x))
    to_small_int = lambda x: all_indices_list.index(x)
    expr = ','.join(
        ''.join(CHARS[to_small_int(i)] for i in t.indices)
        for t in bucket) +\
            '->'+''.join(CHARS[to_small_int(i)] for i in result_indices)
    return expr




def permute_torch_tensor_data(data:np.ndarray, indices_in, indices_out):
    """
    Permute the data of a numpy tensor to the given indices_out.
    
    Returns:
        permuted data
    """
    # permute indices
    out_locs = {idx: i for i, idx in enumerate(indices_out)}
    perm = [out_locs[i] for i in indices_in]
    # permute tensor
    return torch.permute(data, perm)

def slice_torch_tensor(data:np.ndarray, indices_in, indices_out, slice_dict):
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
    #print(f'{indices_in=}, {indices_sliced=} {slice_dict=}, {slice_bounds=}, slicedix {indices_sliced}, sshape {s_data.shape}')
    indices_sized = [v.copy(size=size) for v, size in zip(indices_sliced, s_data.shape)]
    indices_out = [v for v in indices_out if not isinstance(slice_dict.get(v, None), int)]
    assert len(indices_sized) == len(s_data.shape)
    assert len(indices_sliced) == len(s_data.shape)
    st_data = permute_torch_tensor_data(s_data, indices_sliced, indices_out)
    return st_data, indices_out


class TorchBackend(ContractionBackend):

    def __init__(self, device='cpu'):
        # alias of gpu -> cuda
        if device=='gpu':
            device='cuda'
        # Check that CUDA is available if specified
        if device=='cuda':
            if not torch.cuda.is_available():
                logger.warning("Cuda is not available. Falling back to CPU")
                device = 'cpu'
        if device=='xpu':
            import intel_extension_for_pytorch as ipex

                
        self.device = torch.device(device)
        logger.debug("Torch backend using device {}", self.device)
        self.dtype = ['float', 'double', 'complex64', 'complex128']
        self.width_dict = [set() for i in range(30)]
        self.width_bc = [[0,0] for i in range(30)] #(#distinct_bc, #bc)

    def process_bucket(self, bucket, no_sum=False):
        bucket.sort(key = lambda x: len(x.indices))
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        width = len(set(bucket[0].indices))

        for tensor in bucket[1:-1]:

            expr = get_einsum_expr(
                list(map(int, result_indices)), list(map(int, tensor.indices))
            )

            logger.trace('Before contract. Expr: {}, inputs: {}, {}', expr, result_data, tensor) 
            result_data = torch.einsum(expr, result_data, tensor.data)
            logger.trace("expression {}. Data: {}, -> {}", expr, tensor.data, result_data)

            # Merge and sort indices and shapes
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int, reverse=True
            )
            )
            
            size = len(set(tensor.indices))
            if size > width:
                width = size

            self.width_dict[width].add(expr)
            self.width_bc[width][0] = len(self.width_dict[width])
            self.width_bc[width][1] += 1

        if len(bucket)>1:
            tensor = bucket[-1]
            expr = get_einsum_expr(
                list(map(int, result_indices)), list(map(int, tensor.indices))
                , contract = 0 if no_sum else 1
            )
            logger.trace('Before contract. Expr: {}, inputs: {}, {}', expr, result_data, tensor) 
            result_data = torch.einsum(expr, result_data, tensor.data)
            logger.trace("expression {}. Data: {}, -> {}", expr, tensor.data, result_data)
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int, reverse=True
            ))
        else:
            if not no_sum:
                result_data = result_data.sum(axis=-1)
            else:
                result_data = result_data


        if len(result_indices) > 0:
            first_index = result_indices[-1]
            if not no_sum:
                result_indices = result_indices[:-1]
            tag = first_index.identity
        else:
            tag = 'f'
            result_indices = []

        # reduce
        result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                            data=result_data)
        return result

    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        result_indices = bucket[0].indices
        # print("result_indices", result_indices)
        result_data = bucket[0].data
        all_indices = set(sum((list(t.indices) for t in bucket), []))
        result_indices = all_indices - set(ixs)
        all_indices_list = list(all_indices)
        to_small_int = lambda x: all_indices_list.index(x)
        tensors = []
        is128 = False
        for tensor in bucket:
            if tensor.data.dtype in [torch.float64]:
                tensors.append(tensor.data.type(torch.complex64))
            else:
                tensors.append(tensor.data)
            if tensor.data.dtype == torch.complex128:
                is128 = True
        
        if is128:
            for i in range(len(tensors)):
                tensors[i] = tensors[i].type(torch.complex128)
        
        expr = get_einsum_expr_bucket(bucket, all_indices_list, result_indices)
        expect = len(result_indices)
        result_data = torch.einsum(expr, *tensors)

        if len(result_indices) > 0:
            first_index, *_ = result_indices
            tag = str(first_index)
        else:
            tag = 'f'

        result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                            data=result_data)
        
        return result

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        sliced_buckets = []
        for bucket in buckets:
            sliced_bucket = []
            for tensor in bucket:
                # get data
                # sort tensor dimensions
                out_indices = list(sorted(tensor.indices, key=int, reverse=True))
                if tensor.data is None:
                    data = data_dict[tensor.data_key]
                else:
                    data = tensor.data
                # Works for torch tensors just fine
                if not isinstance(data, torch.Tensor):             
                    data = torch.from_numpy(data.astype(np.complex128)).to(self.device)
                else:
                    data = data.type(torch.complex128)
                # slice data
                data, new_indices = slice_torch_tensor(data, tensor.indices, out_indices, slice_dict)

                sliced_bucket.append(
                    tensor.copy(indices=new_indices, data=data))
            sliced_buckets.append(sliced_bucket)

        return sliced_buckets

    def get_result_data(self, result):
        return torch.permute(result.data, tuple(reversed(range(result.data.ndim))))

class TorchBackendMatm(TorchBackend):

    def _get_index_sizes(self, *ixs, size_dict = None):
        if size_dict is not None:
            return [size_dict[i] for i in ixs]
        try:
            sizes = [ i.size for i in ixs ]
        except AttributeError:
            sizes = [2] * len(ixs)
        return sizes

    def _get_index_space_size(self, *ixs, size_dict = None):
        sizes = self._get_index_sizes(*ixs, size_dict = size_dict)
        return reduce(np.multiply, sizes, 1)

    def pairwise_sum_contract(self, ixa, a, ixb, b, ixout, size_dict = None):
        out = ixout
        common = set(ixa).intersection(set(ixb))
        # -- sum indices that are in one tensor only
        all_ix = set(ixa+ixb)
        sum_ix = all_ix - set(out)
        a_sum = sum_ix.intersection(set(ixa) - common)
        b_sum = sum_ix.intersection(set(ixb) - common)
        #print('ab', ixa, ixb)
        #print('all sum', sum_ix, 'a/b_sum', a_sum, b_sum)
        if len(a_sum):
            a = a.sum(axis=tuple(ixa.index(x) for x in a_sum))
            ixa = [x for x in ixa if x not in a_sum]
        if len(b_sum):
            b = b.sum(axis=tuple(ixb.index(x) for x in b_sum))
            ixb = [x for x in ixb if x not in b_sum]
        tensors = a, b
        # --

        ixs = ixa, ixb
        common = set(ixs[0]).intersection(set(ixs[1]))

        # \sum_k A_{kfm} * B_{kfn} = C_{fmn}
        mix = set(ixs[0]) - common
        nix = set(ixs[1]) - common
        kix = common - set(out)
        fix = common - kix
        common = list(kix) + list(fix)
        #print(f'{ixa=} {ixb=} {ixout=}; {common=} {mix=} {nix=}, {size_dict=}')
        if tensors[0].numel() > 1:
            a = tensors[0].permute(*[
                list(ixs[0]).index(x) for x in common + list(mix)
            ])

        if tensors[1].numel() > 1:
            b = tensors[1].permute(*[
                list(ixs[1]).index(x) for x in common + list(nix)
            ])

        k, f, m, n = [self._get_index_space_size(*ix, size_dict=size_dict)
                      for ix in (kix, fix, mix, nix)
                     ]
        a = a.reshape(k, f, m)
        b = b.reshape(k, f, n)
        c = torch.einsum('kfm, kfn -> fmn', a, b)
        if len(out):
            #print('out ix', out, 'kfmnix', kix, fix, mix, nix)
            c = c.reshape(*self._get_index_sizes(*out, size_dict=size_dict))
        #print('outix', out, 'res', c.shape, 'kfmn',kix, fix, mix, nix)

        current_ord_ = list(fix) + list(mix) + list(nix)
        if len(out):
            c = c.permute(*[current_ord_.index(i) for i in out])
        else:
            c = c.flatten()
        #print(f'c shape {c.shape}')
        return c

    def process_bucket(self, bucket, no_sum=False):
        bucket.sort(key = lambda x: len(x.indices))
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        width = len(set(bucket[0].indices))


        for tensor in bucket[1:-1]:

            ixr = list(map(int, result_indices))
            ixt = list(map(int, tensor.indices))
            out_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int, reverse=True
            )
            )
            ixout = list(map(int, out_indices))

            logger.trace('Before contract. expr: {}, {} -> {}', ixr, ixt, ixout)
            size_dict = {}
            for i in result_indices:
                size_dict[int(i)] = i.size
            for i in tensor.indices:
                size_dict[int(i)] = i.size
            logger.debug("result_indices: {}", result_indices)
            result_data_new = self.pairwise_sum_contract(ixr, result_data, ixt, tensor.data, ixout, size_dict = size_dict)
            result_indices = out_indices
            #result_data = torch.einsum(expr, result_data, tensor.data)
            logger.trace("Data: {}, {} -> {}", result_data.shape, tensor.data.shape, result_data_new.shape)
            result_data = result_data_new

            # Merge and sort indices and shapes
            
            size = len(set(tensor.indices))
            if size > width:
                width = size


        if len(bucket)>1:
            tensor = bucket[-1]

            ixr = list(map(int, result_indices))
            ixt = list(map(int, tensor.indices))
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int, reverse=True
            ))[:-1]
            ixout = list(map(int, result_indices))

            logger.trace('Before contract. expr: {}, {} -> {}', ixr, ixt, ixout)
            size_dict = {}
            for i in result_indices:
                size_dict[int(i)] = i.size
            for i in tensor.indices:
                size_dict[int(i)] = i.size
            #logger.debug("result_indices: {}", result_indices)
            result_data_new = self.pairwise_sum_contract(ixr, result_data, ixt, tensor.data, ixout, size_dict = size_dict)
            #result_data = torch.einsum(expr, result_data, tensor.data)
            logger.trace("Data: {}, {} -> {}", result_data.mean(), tensor.data.mean(), result_data_new.mean())
            #if result_data_new.mean() == 0:
            #    logger.warning("Result is zero")
            #    logger.debug("result_indices: {}", result_indices)
            #    logger.debug("result_data: {}", result_data)
            #    logger.debug("tensor: {}", tensor)
            #    logger.debug("tensor_data: {}", tensor.data)
            #    logger.debug("result_data_new: {}", result_data_new)
            #    raise ValueError("Result is zero")
            result_data = result_data_new
        else:
            result_data = result_data.sum(axis=-1)
            result_indices = result_indices[:-1]

        if len(result_indices) > 0:
            first_index = result_indices[-1]
            tag = first_index.identity
        else:
            tag = 'f'
            result_indices = []

        # reduce
        result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                            data=result_data)
        return result


