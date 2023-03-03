"""
Run `watch -n 0.1 nvidia-smi` and then run this test
"""
from qtensor.compression import CUSZCompressor
import cupy
import ctypes

def free_compressed(ptr):
    cmp_bytes, *_ = ptr
    p_decompressed_ptr = ctypes.addressof(cmp_bytes)
    # cast to int64 pointer
    # (effectively converting pointer to pointer to addr to pointer to int64)
    p_decompressed_int= ctypes.cast(p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64))
    decompressed_int = p_decompressed_int.contents
    cupy.cuda.runtime.free(decompressed_int.value)

def test_leak():
    N = 1024*1024*8 # 64MB
    a = cupy.zeros(N, dtype=float)
    a[::1024] = .1

    c = CUSZCompressor()
    for i in range(100):
        out = c.compress(a)
        print(i, "Compressed size", c.compress_size(out)/1024**2, "MB")
        b = c.decompress(out)
        print(i, "Decompressed, 0, 1024", b[0], b[1024])
        c.free_decompressed()
        free_compressed(out)
