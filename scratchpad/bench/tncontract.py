from types import DynamicClassAttribute
from numpy.core.fromnumeric import shape, size
import pyrofiler
from typing import List
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
import itertools
import platform
import importlib
import os, psutil
from cupy import cutensor as cupy_cutensor

from base import LasyModule, BenchResult, Backend, get_gpu_props_json

np = LasyModule('numpy')
torch = LasyModule('torch')
cupy = LasyModule('cupy')

import sys
import os
sys.path.append(os.environ['HOME']+"/.local")
exatn = LasyModule('exatn')

class Tncontract(Backend):
    @staticmethod
    def get_task_type():
        return "tncontract"
    
    @classmethod
    def get_params(cls, *sizes):
        ops = np.prod(sizes[0]) * sizes[1][3] 
        param_in = np.prod(sizes[0]) + np.prod(sizes[1])
        param_out = sizes[0][0] * sizes[0][2] * sizes[1][3]
        return ops.item(), param_in.item(), param_out


class Numpy(Tncontract):
    @staticmethod
    def get_dtype(dtype):
        return {
            'float':np.float32
            ,'double': np.float64
            ,'complex64': np.complex64
            ,'complex128': np.complex128
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return np.random.rand(*sizes).astype(dtype_t)
    
    @staticmethod
    def get_operation():
        return np.einsum
    


class Torch(Tncontract):
    torch.backends.cuda.matmul.allow_tf32 = False
    gpu_tensor = ['float', 'double']

    @staticmethod
    def get_dtype(dtype):
        return {
            'float':torch.cuda.FloatTensor
            ,'double': torch.cuda.DoubleTensor
            ,'complex64': torch.complex64
            ,'complex128': torch.complex128
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        if dtype in cls.gpu_tensor:
            dtype = cls.get_dtype(dtype)
            return dtype(*sizes).normal_()
        else:
            dtype = cls.get_dtype(dtype)
            return torch.rand(*sizes, dtype=dtype, device='cuda')
    
    @staticmethod
    def get_operation():
        return torch.einsum

class TorchCuda(Torch):
    @classmethod
    @contextmanager
    def timing(cls, **kwargs):
        class Foo:
            pass
        res = Foo()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield res
        end.record()
        torch.cuda.synchronize()
        res.result = start.elapsed_time(end)/1000


class Cupy(Tncontract):
    @classmethod
    @contextmanager
    def timing(cls, **kwargs):
        class Foo:
            pass
        res = Foo()
        start = cupy.cuda.Event(disable_timing=False)
        end = cupy.cuda.Event(disable_timing=False)
        start.record()
        yield res
        end.record()
        end.synchronize()
        res.result = cupy.cuda.get_elapsed_time(start, end)/1000

    @staticmethod
    def get_dtype(dtype):
        return {
            'float':cupy.float32
            ,'double': cupy.float64
            ,'complex64': cupy.complex64
            ,'complex128': cupy.complex128
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return cupy.random.rand(*sizes).astype(dtype_t)
    
    @staticmethod
    def get_operation():
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cupy.einsum


class CuTensor(Cupy):
    @classmethod
    def get_ready(self, num_tensors, *sizes):
        sizes = list(sizes)
        num_tensors += 1
        unit_size = sizes[0][0]
        sizes.append([unit_size for i in range(3)])
        return num_tensors, *sizes

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return cupy.random.rand(*sizes).astype(dtype_t)

    @classmethod
    def tncontract(cls, contraction, *tensors):
        [x, desc_x] = tensors[0]
        [y, desc_y] = tensors[1]
        [z, desc_z] = tensors[2]
        return cupy_cutensor.contraction(1.0, x, desc_x, ('A', 'B', 'C', 'D'), 
                        y, desc_y, ('B', 'C', 'D', 'F'), 0, 
                        z, desc_z, ('A', 'C', 'F'))
    
    @classmethod
    def get_operation(cls):
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cls.tncontract
    
    @classmethod
    def prepare(cls, x):
        desc_x = cupy_cutensor.create_tensor_descriptor(x)
        return [x, desc_x]



def main():

    experiment_group = "Angela_nslb_tncontract_test"

    contraction = 'abcd,bcdf->acf' # tensor

    # Backend
    backends = {
        'numpy':Numpy
        # , 'exatn': Exatn
    }
    if get_gpu_props_json():
        backends.update({
            'torch':TorchCuda
            , 'cupy':Cupy
            , 'cutensor': CuTensor
        })
    
    # Tensor properties
    num_tensors = 2
    dim = 4 # tensor
    # sizes = [2, 4, 8, 10, 16, 20, 30, 32, 40, 50, 60, 64, 70, 80, 100, 120, 128, 130, 150]  # tensor
    sizes = [2, 4, 8, 10, 16, 20, 30, 32, 40, 50, 60, 64, 70, 80, 100, 120]  # tensor
    dtypes = ['float', 'double', 'complex64', 'complex128']

    # Test properties
    repeats = 5
    use_strip = True
    if use_strip:
        repeats += 2
    
    # Bechmark
    for backend in backends:
        for size in sizes:
            input_sizes = [size for i in range(dim)] # square tensors
            size = [input_sizes, input_sizes]
            results = []
            for dtype in dtypes:
                for _ in range(repeats):
                    b = backends[backend]
                    _, bench_result = b.benchmark(num_tensors, *size, dtype=dtype, contraction=contraction)
                    results.append(bench_result)
                json_result = b.print_results_json(use_strip, backend, *size, dtype=dtype, results=results, experiment_group=experiment_group)
                          


if __name__ == "__main__":
    main()

