from numpy.core.fromnumeric import shape, size
import pyrofiler
from typing import List
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import itertools
import platform
import importlib
import os, psutil

class LasyModule:
    def __init__(self, modulename):
        self.modulename = modulename
        self.module = None
    def __getattr__(self, attr):
        if self.module is None:
            self.module = importlib.import_module(self.modulename)
        return self.module.__getattribute__(attr)

np = LasyModule('numpy')
torch = LasyModule('torch')
cupy = LasyModule('cupy')
opt_einsum = LasyModule('opt_einsum')

from cupy import cutensor as cupy_cutensor

import sys
import os
sys.path.append(os.environ['HOME']+"/.local")
exatn = LasyModule('exatn')

@dataclass
class BenchResult:
    gen_time: float
    transfer_time: float
    mult_time: float

class Backend:
    @staticmethod
    def prepare(x):
        return x


    @staticmethod
    def get_result(x):
        return x

    timing=pyrofiler.timing

    @classmethod
    def gen_tensor(cls, size_x, size_y, dtype='float'):
        raise NotImplementedError

    @classmethod
    def get_operation(cls, task_type):
        if task_type == "matmul":
            return cls.get_matmul()
        elif task_type == "tncontract":
            return cls.get_tncontract()

    @staticmethod
    def get_matmul():
        raise NotImplementedError

    @staticmethod
    def get_tncontract():
        raise NotImplementedError


    @classmethod
    def benchmark(cls, task_type, size_x, size_y, dtype, contraction=''):
        # this line will also trigger lazy import
        import psutil
        process = psutil.Process(os.getpid())
        overhead = process.memory_info().rss
        # print(overhead)
        operation = cls.get_operation(task_type)
        
        with cls.timing(callback=lambda x: None) as gen:
            x = cls.gen_tensor(*size_x, dtype=dtype)
            y = cls.gen_tensor(*size_y, dtype=dtype)
            if cls == CuTensor:
                # size_z = tuple([size_x[0], size_x[2],size_y[2]])
                size_z = tuple([size_x[0], size_x[2],size_y[3]])
                z = cls.gen_tensor(*size_z, dtype=dtype)
        with cls.timing(callback=lambda x: None) as prep:
            x = cls.prepare(x)
            y = cls.prepare(y)
            if cls == CuTensor:
                z = cls.prepare(z)
        with cls.timing(callback=lambda x: None) as mm:
            if cls == CuTensor:
                z = operation(x,y,z)
            else:
                if task_type == "tncontract":
                    z = operation(contraction,x,y)
                elif task_type == "matmul":
                    z = operation(x,y)
        with cls.timing(callback=lambda x: None) as get:
            zr = cls.get_result(z)
        return zr, BenchResult(gen_time=gen.result, transfer_time=prep.result+get.result, mult_time=mm.result)

class Numpy(Backend):
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
    def get_matmul():
        return np.matmul
    
    @staticmethod
    def get_tncontract():
        return np.einsum

class OptEinsum(Numpy):
    @staticmethod
    def get_matmul():
        raise NotImplementedError

    @staticmethod
    def get_tncontract():
        return opt_einsum.contract

class Torch(Backend):
    @staticmethod
    def get_dtype(dtype):
        return {
            'float':torch.float32
            ,'double': torch.float64
            ,'complex64': torch.complex64
            ,'complex128': torch.complex128
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype = cls.get_dtype(dtype)
        return torch.rand(*sizes, dtype=dtype)

    @staticmethod
    def get_matmul():
        return torch.matmul
    
    @staticmethod
    def get_tncontract():
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

    @staticmethod
    def prepare(x):
        return x.to('cuda')

class Cupy(Backend):
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

        #I'm not sure about this line, just guessed by analogy from torch
        # Without it raises DeviceNotReady erorr
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
    def get_matmul():
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cupy.matmul
    
    @staticmethod
    def get_tncontract():
        with pyrofiler.timing('cblas handler'): #??
            cupy.cuda.device.get_cublas_handle()
        return cupy.einsum

class CuTensor(Cupy):
    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return cupy.random.rand(*sizes).astype(dtype_t)

    @classmethod
    def tncontract(cls, x, y, z):
        [x, desc_x] = x
        [y, desc_y] = y
        [z, desc_z] = z
        # return cupy_cutensor.contraction(1, x, desc_x, cls.mode_x, 
        #                         y, desc_y, cls.mode_y, 0, 
        #                         z, desc_z, cls.mode_z)
        return cupy_cutensor.contraction(1.0, x, desc_x, ('A', 'B', 'C', 'D'), 
                        y, desc_y, ('B', 'C', 'D', 'F'), 0, 
                        z, desc_z, ('A', 'C', 'F'))
    
    @classmethod
    def get_matmul(cls):
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cls.tncontract
    
    @classmethod
    def get_tncontract(cls):
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cls.tncontract
    
    @classmethod
    def prepare(cls, x):
        if not hasattr(cls, 'mode_x'):
            cls.mode_x = ('a', 'b', 'c', 'd')
            # cls.mode_y = ('c', 'd', 'f', 'e')
            cls.mode_y = ('b', 'c', 'd', 'f')
            cls.mode_z = ('a', 'c', 'f')
            cls.mode_x = cupy_cutensor.create_mode(*cls.mode_x)
            cls.mode_y = cupy_cutensor.create_mode(*cls.mode_y)
            cls.mode_z = cupy_cutensor.create_mode(*cls.mode_z)
        desc_x = cupy_cutensor.create_tensor_descriptor(x)
        return [x, desc_x]

def format_flops(flops):
    ord = 3*int(np.log10(flops)/3)
    suffix = {
        3: 'k'
        ,6: 'M'
        ,9: 'G'
        , 12: 'T'
    }[ord]
    return f'{(flops/10**ord).round(2)}{suffix}'

def obj2dict(obj):
    keys = [x for x in dir(obj) if x[0]!='_']
    return dict((key, obj.__getattribute__(key)) for key in keys)

@lru_cache
def get_gpu_props_json():
    try:
        import torch
        devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
        return obj2dict(devprops)
    except:
        return None

def get_dtype_size(dtype):
    dtype_t = Numpy.get_dtype(dtype)
    x = np.ones(10, dtype=dtype_t)
    return x.itemsize


def mean_mmax(x: list):
    mx, mn = max(x), min(x)
    x.remove(mx)
    x.remove(mn)
    return np.mean(x)

#whether to use the removal of max and min before mean
# does not affect standard deviation or other times, only matmul
use_strip = True

def print_results_json(task_type, backend, size_x, size_y, dtype, results: List[BenchResult], experiment_group="default group"):
    import json
    GPU_PROPS = get_gpu_props_json()
    tt1 = [r.gen_time for r in results]
    tt2 = [r.mult_time for r in results]
    tt3 = [r.transfer_time for r in results]
    m1, m3 = np.mean(tt1), np.mean(tt3)
    if use_strip:
        m2 = mean_mmax(tt2)
    else:
        m2 = np.mean(tt2)
    s1, s2, s3 = np.std(tt1), np.std(tt2), np.std(tt3)
    if task_type == "matmul":
        ops = np.prod(size_x) * size_y[1]
        size = [x for x in size_x]
        size.append(size_y[1])
        bytes = get_dtype_size(dtype)*ops.item()
    elif task_type == "tncontract":
        ops = np.prod(size_x) * size_y[3] # Hardcode
        size = size_x[0]
        bytes = get_dtype_size(dtype)*size**6 # Hardcode

        # ops = np.prod(size_x) * np.prod(size_y[2:4]) # Hardcode
        # size.append(size_y[2].item())
        # bytes = get_dtype_size(dtype)*size_x[0].item()*size_x[2].item()*size_y[2].item() # Hardcode
        

    ops = ops.item()
    flops = ops/m2

    res = dict(
        task_type=task_type
        , backend=backend
        , size=size
        , itemsize=get_dtype_size(dtype)
        # , overhead=overhead
        , bytes=bytes 
        , dtype=dtype
        , device_props=dict(name=platform.node(), gpu=GPU_PROPS)
        , transfer_time=m3
        , transfer_relstd=s3
        , gen_time=m1
        , gen_relstd=s1
        , mult_time=m2
        , mult_relstd=s2
        , ops=ops
        , flops=flops
        , flops_str=format_flops(flops)
        , experiment_group=experiment_group
        , fbratio=ops/bytes
    )
    print(json.dumps(res), flush=True)
    return res


def main():
    # args
    # task_type = "matmul"
    task_type = "tncontract"
    experiment_group = "Angela_pod_tncontract_fake_test_fixed_size"
    num_size = 10
    max_size = 64 #64
    # contraction = 'abcd,cdfe->acf'
    contraction = 'abcd,bcdf->acf'
    repeats = 5
    if use_strip:
        repeats += 2
    
    if task_type == "matmul":
        sizes_x = [
            [10, 10], [100, 100], [1000, 1000], [1024, 1024], [1025, 1025],
            [2000, 2000], [4090, 4090], [4096, 4096]
        ]
        sizes_y = [
            [10, 10], [100, 100], [1000, 1000], [1024, 1024], [1025, 1025],
            [2000, 2000], [4090, 4090], [4096, 4096]
        ]
    elif task_type == "tncontract":
        sizes_x = []
        sizes_y = []

        for size in [2, 4, 8, 10, 16, 20, 32, 40, 60, 64, 70, 100, 128, 250, 256, 300]:
            sizes_x.append([size for i in range(4)])
            sizes_y.append([size for i in range(4)])
        
        # for i in range(num_size):
        #     sizes = list(np.random.randint(1,max_size+1,size=6))
        #     sizes_x.append(tuple(sizes[0:4])) # a b c d
        #     # sizes_y.append(tuple(sizes[2:6]))
        #     sizes_y.append(tuple(sizes[1:5])) # b c d e


    backends = {
        'numpy':Numpy
        , 'opt_einsum':OptEinsum
        # , 'exatn': Exatn
    }
    if get_gpu_props_json():
        backends.update({
            'torch':TorchCuda
            , 'cupy':Cupy
            , 'cutensor': CuTensor
        })

    dtypes = ['float', 'double', 'complex64', 'complex128']
    
    import json

    with open('data.json', mode='w') as f:
        for backend in backends:
            for [size_x, size_y] in zip(sizes_x, sizes_y):
                results = []
                for dtype in dtypes:
                    for _ in range(repeats):
                        b = backends[backend]
                        _, bench_result = b.benchmark(task_type, size_x, size_y, dtype=dtype, contraction=contraction)
                        results.append(bench_result)
                    json_result = print_results_json(task_type, backend, size_x, size_y, dtype, results, experiment_group)
                    f.write(json.dumps(json_result))
                    f.write(",")

if __name__ == "__main__":
    main()

