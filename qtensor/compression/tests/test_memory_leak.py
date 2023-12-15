"""
Run `watch -n 0.1 nvidia-smi` and then run this test
"""
from qtensor.compression import CUSZXCompressor
import cupy
import ctypes


def _init_nvsmi():
    import nvidia_smi
    nvidia_smi.nvmlInit()
    nvsmi_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    return nvsmi_handle

def _get_nvsmi_mem(handle):
    import nvidia_smi
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    mem = info.used
    return mem


def test_leak():
    dtype = cupy.complex64
    dtype_size = dtype(0).nbytes
    MB_elems = int(1024 ** 2 / dtype_size)
    MB_target = 128
    N = MB_target * MB_elems
    print(f"== Testing memory leak with {N} elements and {MB_target} MB array ==")

    a = cupy.zeros(N, dtype=dtype)
    a[::1024] = 0.01
    a[::8] = cupy.random.rand(N // 8) * 0.01
    for i in range(1000):
        a[32 * i] = 0.005 * (i % 5 + 1)
    _nvsmi_handle = _init_nvsmi()
    print(f"Original, [0]={a[0]}, [1024]={a[1024]}")

    c = CUSZXCompressor(r2r_error=1e-2, r2r_threshold=1e-2)
    for i in range(100):
        out = c.compress(a)
        print(i, "Compression ratio", 4 * N / c.compress_size(out))
        b = c.decompress(out)
        a[:] = b
        print(i, f"Decompressed, [0]={b[0]}, [1024]={b[1024]}")
        c.free_decompressed()
        c.free_compressed(out)
        print(f"== [{i}] Memory usage: {_get_nvsmi_mem(_nvsmi_handle) / 1024 ** 3} GB ==")
