# Exatn tutorial

# %%
import sys
sys.path.append("/home/plate/.local/lib")
import exatn
import numpy as np
#%%
x, y, z = [np.random.randn(*sh) for sh in [
    (2, 3),
    (3, 2, 2),
    (2, 2, 2)
]]
# %% [markdown]
"""
## Set up exatn tensors
Turns out exatn modifies the original arrays, so have to use copy
"""
# %%
exatn.createTensor('X', np.array(x, copy=True))
exatn.createTensor('Y', np.array(y, copy=True))
exatn.createTensor('Z', np.array(z, copy=True))

# %%
tn = exatn.TensorNetwork('test')
tn.appendTensor(1, 'X')
tn.appendTensor(2, 'Y', [(1, 0)])
tn.appendTensor(3, 'Z', [(1, 0), (2,1)])
#%%
tn.printIt()
# %%
exatn.evaluate(tn)
# %%
result = tn.getTensor(0)
# %%
result_name = result.getName()
result_data = exatn.getTensorData(result_name)
# %%
result_data

# %%
x_t = x.transpose((1,0))
y_t = y.transpose((2,1,0))
z_t = z.transpose((2,1,0))
#%%
x, x_t
#%%
einsum_data = np.einsum('ij,jkl,klm->im', x, y, z)
einsum_data = np.einsum('ji,lkj,mlk->mi', x_t, y_t, z_t)
# %% [markdown]
"""
## Compare results to einsum
"""
# %%
assert np.allclose(einsum_data, result_data),\
f"Numpy result:\n {einsum_data} \n != exatn result: \n {result_data}"
# %% [markdown]
"""
## Compare input data
"""
# %%
exatn_x = exatn.getTensorData('X')
assert np.allclose(exatn_x, x), f"numpy: {x} != exatn: {exatn_x}"
# %% [markdown]
"""
## Run using `evaluateTensorNetwork`
"""

#%%
exatn.createTensor('F0', [2, 2], 0.)
# %%
exatn.createTensor('X0', np.array(x, copy=True))
exatn.createTensor('Y0', np.array(y, copy=True))
exatn.createTensor('Z0', np.array(z, copy=True))
# %%
exatn.evaluateTensorNetwork('test2', 'F0(a,b) = X0(a,c) * Y0(c,d,e) * Z0(d,e,b)')
# %%
eval_data = exatn.getLocalTensor('F0')
eval_data
# %%
assert np.allclose(eval_data, result_data)

# %%

# %% [markdown]
"""
## Try to check matrix multiplication

Use different dimensions for each axis to be sure that transpositions are correct.
"""
# %%
N = 2
a, b = np.random.randn(N, N+1), np.random.randn(N+1, N+2)

exatn.createTensor('C1', [N, N+2], 0.)
exatn.createTensor('A1', np.array(a, copy=True))
exatn.createTensor('B1', np.array(b, copy=True))
# %%
exatn.evaluateTensorNetwork('test3', 'C1(a, c) = A1(a, b) * B1(b, c)')
# %%
c1_exatn = exatn.getLocalTensor('C1')
# %%
assert np.allclose(c1_exatn, a.dot(b))
# %%
a.dot(b)
# %%
c1_exatn
# %%
