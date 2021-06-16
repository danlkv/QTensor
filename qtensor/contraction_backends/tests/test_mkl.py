import pytest
import numpy as np
import qtensor
tcontract = pytest.importorskip('tcontract')

def set_env_mkl():
    """ to use mkl contraction one has to tell
    where to look for dynamic libraries.
    This approach won't work on my system,
    so you can try running exports before running the
    test
    """
    import os
    MKL='/soft/compilers/intel-2020/compilers_and_libraries/linux/mkl/lib/intel64_lin'

    os.environ['LD_PRELOAD'] = f'{MKL}/libmkl_def.so:{MKL}/libmkl_avx2.so:{MKL}/libmkl_core.so:{MKL}/libmkl_sequential.so:{MKL}/libmkl_intel_thread.so:{MKL}/libmkl_gnu_thread.so:{MKL}/libmkl_rt.so'

    os.environ['LD_LIBRARY_PATH']= f'/soft/compilers/intel-2020/compilers_and_libraries/linux/mkl/lib/intel64_lin:{os.environ.get("LD_LIBRARY_PATH")}'


def char2int(x):
    return ord(x) - ord('A')

def test_pairwise():
    backend = qtensor.contraction_backends.CMKLExtendedBackend()

    a = np.random.randn(*[2]*4)
    b = np.random.randn(*[2]*4)
    ai = 'acde'
    bi = 'abcf'
    out= 'cdf'
    c = backend.pairwise_sum_contract(ai, a, bi, b, out)

    ai = list(map(char2int, ai))
    bi = list(map(char2int, bi))
    out = list(map(char2int, out))
    c_ref = np.einsum(a, ai, b, bi, out)
    assert np.allclose(c, c_ref)

def test_pairwise_scalar_out():
    backend = qtensor.contraction_backends.CMKLExtendedBackend()

    a = np.random.randn(*[2]*4)
    b = np.random.randn(*[2]*4)
    ai = 'acde'
    bi = 'abcf'
    out = []
    c = backend.pairwise_sum_contract(ai, a, bi, b, out)

    ai = list(map(char2int, ai))
    bi = list(map(char2int, bi))
    out = list(map(char2int, out))
    c_ref = np.einsum(a, ai, b, bi, out)
    assert np.allclose(c, c_ref)
