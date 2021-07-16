from numpy.core.fromnumeric import size
import pyrofiler
from typing import List
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import itertools
import platform
import importlib

from base import LasyModule, BenchResult, Backend, get_gpu_props_json

np = LasyModule('numpy')
torch = LasyModule('torch')
cupy = LasyModule('cupy')
cupy_cutensor = LasyModule('cutensor')
import sys
import os
sys.path.append(os.environ['HOME']+"/.local")
exatn = LasyModule('exatn')


class Matmul(Backend):    
    @staticmethod
    def get_task_type():
        return "matmul"

    def get_params(*sizes):
        ops = np.prod(sizes[0]) * sizes[1][1]
        param_in = np.prod(sizes[0]) + np.prod(sizes[1])
        param_out = sizes[0][0]*sizes[1][1]
        return ops.item(), param_in.item(), param_out



class Numpy(Matmul):
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
        return np.matmul


class Torch(Matmul):
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
        return torch.matmul


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


class Cupy(Matmul):
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
        return cupy.matmul


class CuTensor(Cupy):
    @classmethod
    def get_ready(self, num_tensors, *sizes):
        sizes = list(sizes)
        num_tensors += 1
        sizes.append([sizes[0][0], sizes[1][1]])
        return num_tensors, *sizes

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return cupy.random.rand(*sizes).astype(dtype_t)

    @classmethod
    def cutensor_matmul(cls, x, y, z):
        [x, desc_x] = x
        [y, desc_y] = y
        [z, desc_z] = z
        from cupy import cutensor
        return cutensor.contraction(1, x, desc_x, cls.mode_x, 
                                y, desc_y, cls.mode_y, 0, 
                                z, desc_z, cls.mode_z)
    
    @classmethod
    def get_operation(cls):
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cls.cutensor_matmul
    
    @classmethod
    def prepare(cls, x):
        from cupy import cutensor
        if not hasattr(cls, 'mode_x'):
            cls.mode_x = ('m', 'k')
            cls.mode_y = ('k', 'n')
            cls.mode_z = ('m', 'n')
            cls.mode_x = cutensor.create_mode(*cls.mode_x)
            cls.mode_y = cutensor.create_mode(*cls.mode_y)
            cls.mode_z = cutensor.create_mode(*cls.mode_z)
        desc_x = cutensor.create_tensor_descriptor(x)
        return [x, desc_x]
    

# @dataclass
# class ExatnTensor:
#     name: str
#     shape: tuple
#     dtype: str

# class Exatn(Backend):
#     # infinite name generator
#     name_generator = (hex(x)[1:] for x, _
#                       in enumerate(itertools.repeat(0)))
#     allocated_tensor_names = []
#     @classmethod
#     def cleanup_tensors(cls):
#         for name in cls.allocated_tensor_names:
#             # will produce a warning on non-existent tensor
#             exatn.destroyTensor(name)
#         cls.allocated_tensor_names = []

#     @staticmethod
#     def get_dtype(dtype):
#         return Numpy.get_dtype(dtype)

#     @classmethod
#     def gen_tensor(cls, *sizes, dtype='float'):
#         tname = next(cls.name_generator)
#         rand = Numpy.gen_tensor(*sizes, dtype=dtype)
#         #print('create ten', tname)
#         success = exatn.createTensor(tname, rand.copy(order='F'))
#         #print('done', tname)
#         if success:
#             cls.allocated_tensor_names.append(tname)
#             return ExatnTensor(name=tname, shape=sizes, dtype=dtype)

#     @classmethod
#     def exatn_matmul(cls, x, y):
#         """
#         Takes two names of tensors, should be already allocated,
#         returns name of resulting tensor

#         Args:
#             x: ExatnTensor
#             y: ExatnTensor
#         """
#         #exatn.evaluateTensorNetwork('sum', 'SR1() = R1(a)*R2(a)')
#         res = next(cls.name_generator)
#         res_shape = x.shape[0], y.shape[1]
#         #print('create res', res, res_shape)
#         dtype = Numpy.get_dtype(x.dtype)
#         res_body = np.zeros(res_shape, dtype=dtype)
#         exatn.createTensor(res, res_body)
#         cls.allocated_tensor_names.append(res)
#         st = f'{res}(a,c)={x.name}(a,b)*{y.name}(b,c)'
#         #print('st', st)
#         _ = exatn.contractTensors(st, 1.0)
#         #print('contr')
#         return res

#     @classmethod
#     def get_operation(cls):
#         return cls.exatn_matmul

#     @classmethod
#     def get_result(cls, x):
#         t = exatn.getLocalTensor(x)
#         cls.cleanup_tensors()
#         return t





def main():

    experiment_group = "Angela_nslb_matmul_test"


    num_tensors = 2
    sizes_m = [10, 100, 1000, 1024, 1025, 2000, 4090, 4096, 8192]
    sizes_n = [10, 100, 1000, 1024, 1025, 2000, 4090, 4096, 8192]
    sizes_l = [10, 100, 1000, 1024, 1025, 2000, 4090, 4096, 8192]
    # sizes_m = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    

    backends = {
        'numpy':Numpy
        # ,'exatn': Exatn
    }
    if get_gpu_props_json():
        backends.update({
            'torch':TorchCuda
            , 'cupy':Cupy
            , 'cutensor': CuTensor
        })

    use_strip = True
    repeats = 5
    if use_strip:
        repeats += 2
    dtypes = ['float', 'double', 'complex64', 'complex128']

    for backend in backends:
        for size_m, size_n, size_l in zip(sizes_m, sizes_n, sizes_l):
            sizes = [size_m,size_n], [size_n, size_l]
            results = []
            for dtype in dtypes:
                for _ in range(repeats):
                    b = backends[backend]
                    _, bench_result = b.benchmark(num_tensors, *sizes, dtype=dtype)
                    results.append(bench_result)
                json_result = b.print_results_json(use_strip, backend, *sizes, dtype=dtype, results=results, experiment_group=experiment_group)

if __name__ == "__main__":
    main()

