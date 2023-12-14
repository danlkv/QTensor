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


def free_compressed(ptr):
    cmp_bytes, *_ = ptr
    p_decompressed_ptr = ctypes.addressof(cmp_bytes[0])
    # cast to int64 pointer
    # (effectively converting pointer to pointer to addr to pointer to int64)
    p_decompressed_int = ctypes.cast(
        p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64)
    )
    decompressed_int = p_decompressed_int.contents
    cupy.cuda.runtime.free(decompressed_int.value)


def test_leak():
    N = 1024 * 1024 * 32  # 32MB
    a = cupy.zeros(N, dtype=float)
    a[::1024] = 0.01
    for i in range(1000):
        a[32 * i] = 0.005 * (i % 5 + 1)
    _nvsmi_handle = _init_nvsmi()
    print(f"Original, [0]={a[0]}, [1024]={a[1024]}")

    c = CUSZXCompressor(r2r_error=1e-2, r2r_threshold=1e-2)
    for i in range(200):
        out = c.compress(a)
        print(i, "Compression ratio", 4 * N / c.compress_size(out))
        b = c.decompress(out)
        #a[:] = b
        print(i, f"Decompressed, [0]={b[0]}, [1024]={b[1024]}")
        c.free_decompressed()
        free_compressed(out)
        print(f"Memory usage: {_get_nvsmi_mem(_nvsmi_handle) / 1024 ** 3} GB")
