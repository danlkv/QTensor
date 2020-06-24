# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts_jupytext//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Sum-of-all-amplitudes-for-all-inputs" data-toc-modified-id="Sum-of-all-amplitudes-for-all-inputs-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Sum of all amplitudes for all inputs</a></span></li><li><span><a href="#Sum-of-amplitudes-for-single-input" data-toc-modified-id="Sum-of-amplitudes-for-single-input-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Sum of amplitudes for single input</a></span></li><li><span><a href="#Single-amplitude-for-single-input" data-toc-modified-id="Single-amplitude-for-single-input-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Single amplitude for single input</a></span></li><li><span><a href="#Simulate-all-output-states-for-given-input" data-toc-modified-id="Simulate-all-output-states-for-given-input-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Simulate all output states for given input</a></span><ul class="toc-item"><li><span><a href="#First-error.-Bucket-order" data-toc-modified-id="First-error.-Bucket-order-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>First error. Bucket order</a></span></li><li><span><a href="#Second-error.-Variable-relabel" data-toc-modified-id="Second-error.-Variable-relabel-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Second error. Variable relabel</a></span></li></ul></li></ul></div>

# +
import numpy as np

import qtree
from qtree.operators import Gate
# -

help(Gate)


# +
class MyGate(Gate):
    name = 'MyGate'
    _changes_qubits=(0,)
    def gen_tensor(self):
        tensor = 1/np.sqrt(2)*np.array([
            [1,1]
            ,[1,-1]
        ])
        return tensor
    
myGate = MyGate(0)
myGate
# -



# +
from qtree import optimizer

tensor_expr, data_dict, bra, ket = optimizer.circ2buckets(1, [[myGate]])
print(tensor_expr)
print(data_dict)
print(bra)
print(ket)
# -

from qtree import np_framework as npfr

# ## Sum of all amplitudes for all inputs
#
# This is just a full contraction of the tensor network

numpy_buckets = npfr.get_sliced_np_buckets(tensor_expr
                                           ,data_dict
                                           ,{}
                                          )
numpy_buckets

result = optimizer.bucket_elimination(numpy_buckets, npfr.process_bucket_np)
result

result.data

# ## Sum of amplitudes for single input
#
# This is a contraction of a network that was sliced over input indices

initial_state = 0
slice_dict = qtree.utils.slice_from_bits(initial_state, ket)
slice_dict

numpy_buckets = npfr.get_sliced_np_buckets(
    tensor_expr, data_dict, slice_dict
)
print(numpy_buckets)
print("Output tensor:",numpy_buckets[0][0].data)
print("Input tensor:", numpy_buckets[-2][0].data)
result = optimizer.bucket_elimination(numpy_buckets, npfr.process_bucket_np)
result.data



# ## Single amplitude for single input

initial_state = 0
output_state = 0
slice_dict = qtree.utils.slice_from_bits(initial_state, ket)
slice_dict.update( qtree.utils.slice_from_bits(output_state, bra) )
slice_dict


def simulate_buckets(tensor_expr
                     , data_dict, slice_dict):
    numpy_buckets = npfr.get_sliced_np_buckets(
        tensor_expr, data_dict, slice_dict
    )
    print(numpy_buckets)
    print("Output tensor:",numpy_buckets[0][0].data)
    print("Input tensor:", numpy_buckets[-2][0].data)
    result = optimizer.bucket_elimination(numpy_buckets, npfr.process_bucket_np)
    return result.data


output = simulate_buckets(tensor_expr, data_dict, slice_dict)
output



# ## Simulate all output states for given input
#
# This is a partial contraction,
# where we leave out the latest, output index

initial_state = 0
slice_dict = qtree.utils.slice_from_bits(initial_state, ket)
#slice_dict.update({var: slice(None) for var in bra})

# ### First error. Bucket order

# +
numpy_buckets = npfr.get_sliced_np_buckets(
    tensor_expr, data_dict, slice_dict
)

print(numpy_buckets)
print("Output tensor:",numpy_buckets[0][0].data)
print("Input tensor:", numpy_buckets[-2][0].data)
print("Input tensor vars:", numpy_buckets[-2][0].indices)
result = optimizer.bucket_elimination(numpy_buckets
                                      , npfr.process_bucket_np
                                      , n_var_nosum=1
                                     )
result.data
# -

# Wrong result! Because the ordering is inverse. We first contracted our input variable, which was already sliced to the first state

# +
reversed_buckets = list(reversed(numpy_buckets))

result = optimizer.bucket_elimination(reversed_buckets
                                      , npfr.process_bucket_np
                                      , n_var_nosum=1
                                     )
result.data
# -

# ### Second error. Variable relabel

# Have to reorder the buckets

all_tensors = sum(tensor_expr, [])
print(all_tensors)
all_vars = set(sum([tensor.indices for tensor in all_tensors], tuple() ))
all_vars

peo = list(all_vars - set(bra))
peo

# +
perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(
    tensor_expr, peo + bra
)

ket_vars = sorted([perm_dict[idx] for idx in ket], key=str)
bra_vars = sorted([perm_dict[idx] for idx in bra], key=str)

initial_state = 0
slice_dict = qtree.utils.slice_from_bits(initial_state, ket_vars)


# +

numpy_buckets = npfr.get_sliced_np_buckets(
    perm_buckets, data_dict, slice_dict
)

print(numpy_buckets)
result = optimizer.bucket_elimination(numpy_buckets
                                      , npfr.process_bucket_np
                                      , n_var_nosum=1
                                     )
result.data
# -


