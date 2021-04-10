import pyrofiler
from typing import List
from contextlib import contextmanager
from dataclasses import dataclass

@dataclass
class BenchResult:
    gen_time: float
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
    def benchmark_matmul(cls, x,y):
        with cls.timing(callback=lambda x: None) as prep:
            x = cls.prepare(x)
            y = cls.prepare(y)
        with cls.timing(callback=lambda x: None) as matmul:
            z = cls.get_matmul()(x,y)
            zr = cls.get_result(z)
        return zr, BenchResult(gen_time=prep.result, mult_time=matmul.result)

class Numpy(Backend):
    @staticmethod
    def get_dtype(dtype):
        import numpy as np
        return {
            'float':np.float32
            ,'double': np.float64
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        import numpy as np
        dtype = cls.get_dtype(dtype)
        return np.random.rand(*sizes).astype(dtype)

    @staticmethod
    def get_matmul():
        import numpy as np
        return np.matmul


class Torch(Backend):
    @staticmethod
    def get_dtype(dtype):
        import torch
        return {
            'float':torch.float32
            ,'double': torch.float64
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        import torch
        dtype = cls.get_dtype(dtype)
        return torch.rand(*sizes, dtype=dtype)

    @staticmethod
    def get_matmul():
        import torch
        return torch.matmul


class TorchCuda(Torch):
    @classmethod
    @contextmanager
    def timing(cls, **kwargs):
        import torch
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
        import cupy
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
        import cupy
        return {
            'float':cupy.float32
            ,'double': cupy.float64
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        import cupy
        dtype = cls.get_dtype(dtype)
        return cupy.random.rand(*sizes, dtype=dtype)

    @staticmethod
    def get_matmul():
        import cupy
        return cupy.matmul


BACKENDS={
    'numpy':Numpy.get_matmul()
    ,'torch':Torch.get_matmul()
    ,'cupy':Torch.get_matmul()
}


import numpy as np

def format_flops(flops):
    ord = 3*int(np.log10(flops)/3)
    suffix = {
        3: 'k'
        ,6: 'M'
        ,9: 'G'
        , 12: 'T'
    }[ord]
    return f'{(flops/10**ord).round(2)}{suffix}'

def print_results(backend, size, results: List[BenchResult]):
    tt1 = [r.gen_time for r in results]
    tt2 = [r.mult_time for r in results]
    m1, m2 = np.mean(tt1), np.mean(tt2)
    s1, s2 = np.std(tt1), np.std(tt2)
    flops = size**3/m2
    print(f'{backend}, {size}, {m1}, {(s1/m1).round(4)}, {m2}, {(s2/m2).round(4)}, {format_flops(flops)}')


def main():

    sizes = [10, 100, 1000, 1024, 2000, 3000, 3001]
    backends = {
        'numpy':Numpy
        ,'torch':TorchCuda
        ,'cupy':Cupy
    }
    repeats = 10

    print(f'backend, size, Time1 mean, Time1 relstd, Time2 mean, Time2 relstd, FLOPs')
    for backend in backends:
        for size in sizes:
            results = []
            for repeat in range(repeats):
                b = backends[backend]
                x = b.gen_tensor(size, size)
                y = b.gen_tensor(size, size)
                res, bench_result = b.benchmark_matmul(x,y)
                results.append(bench_result)

            print_results(backend, size, results)


if __name__ == "__main__":
    main()

