# Exatn tutorial

# %%
import sys
sys.path.append("/home/plate/.local/lib")
import exatn
import numpy as np
#%%
def S(x):
    """ Flatten array,
     then reshape to inverse shape,
     then transpose to match the original shape
     """
    x = x.reshape(*reversed(x.shape))
    x = x.transpose() # reverses order of axis by default
    return x

def inv_S(x):
    """ Flatten array,
     then reshape to inverse shape,
     then transpose to match the original shape
     """
    x = x.transpose() # reverses order of axis by default
    x = x.reshape(*reversed(x.shape))
    return x

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
    return exatn.createTensor(name, (np.array(value, copy=True)))
# %% [markdown]
"""
## Try to run simplest thing, d=1
"""
#%%
exatn.createTensor('R1', np.array([0.1, 0.2]))
exatn.createTensor('R2', np.array([0., 1. ]))
exatn.createTensor('SR1')
#%%

exatn.evaluateTensorNetwork('sum', 'SR1() = R1(a)*R2(a)')
#%%
exatn.getLocalTensor('SR1')
#%%
exatn.destroyTensor('R1')
exatn.destroyTensor('R2')
exatn.destroyTensor('SR1')
#%%
# %% [markdown]
"""
## Try to run next to simplest thing, d=2
"""
#%%
x, y = np.array([[0.2, 0.3], [.12, .67]]), np.array([0., 1.])
#%%
idempotent_tensor_create('R1', x)
idempotent_tensor_create('R2', y)
#%%
exatn.createTensor('SR1', [2], 0.)
#%%

exatn.evaluateTensorNetwork('sum', 'SR1(a) = R1(a,b)*R2(b)')
#%%
r_ex = exatn.getTensorData('SR1')
r_ex
#%%
expr1 = 'ij, j -> i'
r1 = np.einsum(expr1, x, y)
if not np.allclose(r1, r_ex):
    print(f"Expression: `{expr1}` is not equivalent to exatn's")
else:
    print( f"Expression: `{expr1}` is OK")
#%%
expr1 = 'ji, j -> i'
r1 = np.einsum(expr1, x, y)
if not np.allclose(r1, r_ex):
    print( f"Expression: `{expr1}` is not equivalent to exatn's")
else:
    print( f"Expression: `{expr1}` is OK")

# %% [markdown]
"""
## Try to check matrix multiplication

Use different dimensions for each axis to be sure that transpositions are correct.
"""
# %%
#N = 2
#a, b = np.random.randn(N, N+1), np.random.randn(N+1, N+2)
#a, b = [[1., 0], [0., 2.]], [[1., 2.,], [3., 4.]]
#%%
a = np.array([
    [1., 0, 0],
    [0., 1, 1]
], order='F')
b = np.array([
    [1., 0, 3, 0],
    [1,  1,  2, 2],
    [-1, 1, -2, 2],
], order='F')

exatn.createTensor('C1', [2, 4], 0.)
#%%
idempotent_tensor_create('A1', a)
idempotent_tensor_create('B1', b)
# %%
exatn.evaluateTensorNetwork('test3', 'C1(a, c) = A1(a, b) * B1(b, c)')
# %%
c1_exatn = exatn.getLocalTensor('C1')
# %%
try:
    assert np.allclose(c1_exatn, np.dot(a, b)), "direct multiple is not equivalent"
except AssertionError:
    pass
# %%
try:
    assert np.allclose(S(c1_exatn), np.dot(a, b)), "Transformed multiple is not equivalent"
except AssertionError:
    pass
#%%
print('c = a*b\n', np.dot(a, b))
# %%
print('c exatn\n', c1_exatn)
#%%
exatn.destroyTensor('C1')
# %%
# %% [markdown]
"""
## Test more complex tensor network contraction
### Use tn.appendTensor
"""
#%%
x, y, z = [np.random.randn(*sh) for sh in [
    (2, 3),
    (3, 2, 2),
    (2, 2, 2)
]]
# %%
idempotent_tensor_create('X', x)
idempotent_tensor_create('Y', y)
idempotent_tensor_create('Z', z)

# %%
tn = exatn.TensorNetwork('test')
tn.appendTensor(1, 'X')
tn.appendTensor(2, 'Y', [(1, 0)])
tn.appendTensor(3, 'Z', [(1, 0), (2,1)])
#%%
#tn.printItstd()
# %%
exatn.evaluate(tn)
# %%
result = tn.getTensor(0)
# %%
result_name = result.getName()
result_data = exatn.getTensorData(result_name)
# %%
einsum_data = np.einsum('ij,jkl,klm->im', x, y, z)
# %% [markdown]
"""
### Compare results to einsum
"""
# %%
assert np.allclose(einsum_data, result_data),\
f"Numpy result:\n {einsum_data} \n != exatn result: \n {result_data}"
# %% [markdown]
"""
### Compare input data
"""
# %%
exatn_x = exatn.getTensorData('X')
assert np.allclose(exatn_x, x), f"numpy: {x} != exatn: {exatn_x}"
# %% [markdown]
"""
### Run using `evaluateTensorNetwork`
"""

#%%
exatn.createTensor('F0', [2, 2], 0.)
# %%
exatn.createTensor('X0', np.array(x, copy=True))
exatn.createTensor('Y0', np.array(y, copy=True))
exatn.createTensor('Z0', np.array(z, copy=True))
# %%
exatn.evaluateTensorNetwork('test2', 'F0(a,b) = X(a,c) * Y(c,d,e) * Z(d,e,b)')
# %%
eval_data = exatn.getLocalTensor('F0')
eval_data
# %%
assert np.allclose(eval_data, result_data)
#%%
einsum_data
# %% [markdown]
"""
### Fix numerics with reshape-transform
"""
# %%
def S(x):
    """ Flatten array,
     then reshape to inverse shape,
     then transpose to match the original shape
     """
    x = x.reshape(*reversed(x.shape))
    x = x.transpose() # reverses order of axis by default
    return x

def inv_S(x):
    """ Flatten array,
     then reshape to inverse shape,
     then transpose to match the original shape
     """
    x = x.transpose() # reverses order of axis by default
    x = x.reshape(*reversed(x.shape))
    return x
#%%

einsum_data_adj = inv_S(np.einsum('ij,jkl,klm->im', S(x), S(y), S(z)))
#%%
assert np.allclose(einsum_data_adj, eval_data)

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
exatn.evaluateTensorNetwork('test3', 'C1(a, c) = B1(b,c) * A1(a, b)')
# %%
c1_exatn = exatn.getLocalTensor('C1')
# %%
assert np.allclose(c1_exatn, a.dot(b))
# %%
a.dot(b)
# %%
c1_exatn
# %%
exatn.evaluateTensorNetwork('test4', 'C1(a, c) = B1(b,c) * A1(a, b)')

# %%

exatn.getLocalTensor('C1')


#%% [markdown]
"""
## Check tensors generated by exatn
 
"""
# %%
exatn.createTensor('Xr', [2, 3], 0)
exatn.createTensor('Yr', [3, 4], 0)
exatn.initTensorRnd('Xr')
exatn.initTensorRnd('Yr')
# %%
xr = exatn.getTensorData('Xr')
yr = exatn.getTensorData('Yr')
# %%
zr = np.dot(xr, yr)
# %%
exatn.createTensor('Zr', [2, 4], 0)
# %%
exatn.evaluateTensorNetwork('rnd', 'Zr(a, b) = Xr(a, c) * Yr(c, b)')
# %%
zr_exatn = exatn.getTensorData('Zr')
# %%
if not np.allclose(zr_exatn, zr):
    print('Results do not match')
# %% [markdown]
"""
Fix results once again
"""
#%%
t = np.array([[1,0,0],[0, 1, 1]])
S(t)
# %%
inv_S(S(t))
#%%
S(S(S(t)))
#%%
S(np.arange(6).reshape(2, 3))
# %%
adj_zr = inv_S(np.dot(S(xr), S(yr)))
# %%

if not np.allclose(zr_exatn, adj_zr):
    print('Adjusted result also doesnt match')
    print('adj_zr', adj_zr)
    print('exatn', zr_exatn)
else:
    print('Adjusted result matches exatn')
# %%
S(np.array([[0, -1], [1, 0]]))
# %%
S(np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]))
