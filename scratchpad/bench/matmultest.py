import numpy as np
import pyrofiler


dtype = ['float', 'double', 'complex64', 'complex128']
prefix = {
    'float': 2
    ,'double': 2
    ,'complex64': 8
    ,'complex128': 8
}
dtype_t = {
    'float':np.float32
    ,'double': np.float64
    ,'complex64': np.complex64
    ,'complex128': np.complex128
}

sizes = [8192, 8192]
for type in dtype:
    mat_a = np.random.rand(*sizes).astype(dtype_t[type])
    mat_b = np.random.rand(*sizes).astype(dtype_t[type])
    with pyrofiler.timing(callback=lambda x: None) as gen:
        mat_c = np.matmul(mat_a, mat_b)
    print(gen.result)
    ops = 8192 * 8192 * 8192
    flops = prefix[type] * ops / gen.result
    print("dtype:", type)
    print("flops:", flops)