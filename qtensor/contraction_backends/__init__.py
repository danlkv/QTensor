from .base_class import ContractionBackend
from .numpy import NumpyBackend
from .torch import TorchBackend
from .mkl import CMKLExtendedBackend
from .cupy import CuPyBackend
from .transposed import TransposedBackend
from .opt_einsum import OptEinusmBackend
from .performance_measurement_decorator import PerfNumpyBackend
from .transpose_backend import NumpyTranspoedBackend, TorchTransposedBackend, CupyTransposedBackend, CutensorTransposedBackend

def get_backend(name):
    return {
        'einsum': NumpyBackend,
        'torch': TorchBackend,
        'mkl': CMKLExtendedBackend,
        'tr_einsum': NumpyTranspoedBackend,
        'opt_einsum': OptEinusmBackend,
        'tr_torch': TorchTransposedBackend,
        'cupy': CuPyBackend,
        'tr_cupy': CupyTransposedBackend,
        'tr_cutensor': CutensorTransposedBackend
    }[name]()
