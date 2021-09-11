#from torch._C import device
from .base_class import ContractionBackend
from .numpy import NumpyBackend
from .torch import TorchBackend
from .cupy import CuPyBackend
from .mkl import CMKLExtendedBackend
from .cupy import CuPyBackend
from .cutensor import CuTensorBackend
from .transposed import TransposedBackend
from .opt_einsum import OptEinusmBackend
from .transpose_backend import NumpyTranspoedBackend, TorchTransposedBackend, CupyTransposedBackend, CutensorTransposedBackend
from .torch_mix import TorchMixBackend
from .performance_measurement_decorator import PerfNumpyBackend, PerfBackend, GPUPerfBackend
from .mix_decorator import MixBackend

def get_backend(name):
    backend_dict = {
        'mkl': CMKLExtendedBackend,
        'einsum':NumpyBackend,
        'opt_einsum': OptEinusmBackend,
        'torch_cpu': TorchBackend,
        'torch_gpu': TorchBackend,
        'cupy': CuPyBackend,
        'cutensor': CuTensorBackend,
        'tr_einsum': NumpyTranspoedBackend,
        'tr_torch': TorchTransposedBackend,
        'tr_cupy': CupyTransposedBackend,
        'tr_cutensor': CutensorTransposedBackend,
        'torch_mix': TorchMixBackend
    }
    if name == "torch_cpu":
        return backend_dict['torch_cpu'](device = "cpu")
    else:
        return backend_dict[name]()

def get_cpu_perf_backend(name):
    class MyPerfBackend(PerfBackend):
        Backend = {
            'mkl': CMKLExtendedBackend,
            'einsum':NumpyBackend,
            'opt_einsum': OptEinusmBackend,
            'torch_cpu': TorchBackend,     
            'tr_einsum': NumpyTranspoedBackend,
        }[name]

    if name == "torch_cpu":
        return MyPerfBackend(device="cpu")
    else:
        return MyPerfBackend()

def get_gpu_perf_backend(name):
    class MyPerfBackend(GPUPerfBackend):
        Backend = {
            'torch_gpu': TorchBackend,
            'cupy': CuPyBackend,
            'cutensor': CuTensorBackend,
            'tr_torch': TorchTransposedBackend,
            'tr_cupy': CupyTransposedBackend,
            'tr_cutensor': CutensorTransposedBackend
        }[name]

    if name == "torch_gpu":
        return MyPerfBackend(device="gpu")
    else:
        return MyPerfBackend()



# def get_mixed_backend(cpu_name, gpu_name):
#     class MyMixedBackend(MixBackend):
#         CBE = {
#             'einsum':NumpyBackend,
#             'torch_cpu': TorchBackend,
#             'mkl':CMKLExtendedBackend,
#             'opt_einsum': OptEinusmBackend,
#             'tr_einsum': NumpyTranspoedBackend,
#             'opt_einsum': OptEinusmBackend,
#         }[cpu_name]
#         GBE = {
#             'torch_gpu': TorchBackend,
#             'cupy': CuPyBackend,
#             'tr_torch': TorchTransposedBackend,
#             'tr_cupy': CupyTransposedBackend,
#             'tr_cutensor': CutensorTransposedBackend
#         }[gpu_name]
    
#     if cpu_name == "torch_cpu":
#         return MyMixedBackend(device = "cpu")
#     else:
#         return MyMixedBackend()

def get_mixed_backend(cpu_name, gpu_name, threshold = 11):

    cpu_be = get_backend(cpu_name)
    gpu_be = get_backend(gpu_name)
    mix_name = cpu_name + gpu_name
    
    threshold_dict = {
        "einsum-cupy":16,
        "torch_cpu-torch_gpu":11,
        "einsum-torch_gpu":13
    }

    if mix_name in threshold_dict:
        threshold = threshold_dict[mix_name]

    return MixBackend(cpu_be, gpu_be, threshold)


def get_mixed_perf_backend(cpu_name, gpu_name,threshold = 11):
    
    cpu_be = get_cpu_perf_backend(cpu_name)
    gpu_be = get_gpu_perf_backend(gpu_name)
    mix_name = cpu_name + gpu_name
    
    threshold_dict = {
        "einsum-cupy":16,
        "torch_cpu-torch_gpu":11,
        "einsum-torch_gpu":13
    }

    if mix_name in threshold_dict:
        threshold = threshold_dict[mix_name]

    return MixBackend(cpu_be, gpu_be, threshold)

