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
# <div class="toc"><ul class="toc-item"></ul></div>

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

numpy_buckets = npfr.get_sliced_np_buckets(tensor_expr
                                           ,data_dict
                                           ,{}
                                          )
numpy_buckets

result = optimizer.bucket_elimination(numpy_buckets, npfr.process_bucket_np)
result

result.data


