from .base_class import ContractionBackend
from .numpy import NumpyBackend
from .torch import TorchBackend
from .cupy import CuPyBackend
from .mkl import CMKLExtendedBackend
from .transposed import TransposedBackend
from .opt_einsum import OptEinusmBackend
from .performance_measurement_decorator import PerfNumpyBackend

def get_backend(name):
    return {
        'einsum':NumpyBackend,
        'torch': TorchBackend,
        'mkl':CMKLExtendedBackend,
        'tr_einsum': TransposedBackend,
        'opt_einsum': OptEinusmBackend,
        'cupy': CuPyBackend
    }[name]()
