#%%
import sys
from functools import reduce
sys.path.append("/home/plate/.local/lib")
import exatn
import numpy as np
import time

def idempotent_tensor_create(name, value: np.array):
    """ Clears existing tensors before creation,
    protects original arrays from exatn's modification
    """
    if hasattr(exatn, 'tensorAllocated'):
        # tensorAllocated is not included in official bindings, but you can use my fork
        tensor_exists = exatn.tensorAllocated(name)
        if tensor_exists:
            exatn.destroyTensor(name)
    else:
        # will produce a warning on non-existent tensor
        exatn.destroyTensor(name)
    return exatn.createTensor(name, np.array(value, copy=True))
#%%
shapes = [
    (2, 32**2),
    (32**2, 2**6, 24**2),
    (2**6, 24**2, 2),
    (2, 4)
]
tensors = [np.random.randn(reduce(np.multiply, shape, 1)) for shape in shapes]
tensors = [t.reshape(shape, order='F') for shape, t in zip(shapes, tensors)]
idempotent_tensor_create('t1', tensors[0])
idempotent_tensor_create('t2', tensors[1])
idempotent_tensor_create('t3', tensors[2])
idempotent_tensor_create('t5', tensors[3])
#%%
exatn.createTensor('t4', [2, 2], 0.)
exatn.createTensor('t6', [2, 4], 0.)
# %%
exatn.evaluateTensorNetworkAsync('tn1', 't4(a,b)=t1(a,c)*t2(c,d,e)*t3(d,e,b)')
#%%
exatn.evaluateTensorNetworkAsync('tn2', 't6(a,b)=t4(a,c)*t5(c,b)')
# %%
exatn.getLocalTensor('t6')
# %%
start = time.time()
exatn.evaluateTensorNetworkAsync('tn1', 't4(a,b)=t1(a,c)*t2(c,d,e)*t3(d,e,b)')
p1 = time.time()
exatn.evaluateTensorNetworkAsync('tn2', 't6(a,b)=t4(a,c)*t5(c,b)')
p2 = time.time()
tensor_async = exatn.getLocalTensor('t6')
p3 = time.time()

# %%
print(f'async: first net: {p1-start}, second net: {p2-p1}, Get tensor: {p3-p2}')

# %%

start = time.time()
exatn.evaluateTensorNetwork('tn1', 't4(a,b)=t1(a,c)*t2(c,d,e)*t3(d,e,b)')
p1 = time.time()
exatn.evaluateTensorNetwork('tn2', 't6(a,b)=t4(a,c)*t5(c,b)')
p2 = time.time()
tensor_sync = exatn.getLocalTensor('t6')
p3 = time.time()
# %%
print(f'sync: first net: {p1-start}, second net: {p2-p1}, Get tensor: {p3-p2}')

# %%


assert np.allclose(tensor_sync, tensor_async)
# %%
