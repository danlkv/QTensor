"""
Run `watch -n 0.1 nvidia-smi` and then run this test
"""
from qtensor.compression import CUSZXCompressor
import cupy
import numpy as np


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


def test_leak_compress():
    dtype = cupy.complex64
    dtype_size = dtype(0).nbytes
    MB_elems = int(1024**2 / dtype_size)
    MB_target = 128
    N = MB_target * MB_elems
    print(f"== Testing memory leak with {N} elements and {MB_target} MB array ==")
    c = CUSZXCompressor(r2r_error=1e-2, r2r_threshold=1e-2)
    import qtensor

    c = qtensor.compression.ProfileCompressor(c)
    _nvsmi_handle = _init_nvsmi()

    a = cupy.zeros(N, dtype=dtype)
    a[::1024] = 0.01
    a[::8] = cupy.random.rand(N // 8)
    for i in range(1000):
        a[32 * i + 1] = 0.005 * (i % 5 + 1)
    print(f"Original, [0]={a[0]}, [1024]={a[1024]}")

    for j in range(100):
        out = c.compress(a)
        print(i, "Compression ratio", 4 * N / c.compress_size(out))
        b = c.decompress(out)
        # a[:] = b
        print(j, f"Decompressed, [0]={b[0]}, [1024]={b[1024]}")
        c.free_decompressed()
        c.free_compressed(out)
        print(
            f"== [{j}] Memory usage: {_get_nvsmi_mem(_nvsmi_handle) / 1024 ** 3} GB =="
        )


def test_leak_contract():
    from qtensor.compression.CompressedTensor import Tensor
    import qtensor
    from qtree.optimizer import Var
    from qtensor.compression.compressed_contraction import compressed_contract

    dtype = cupy.complex64
    dtype_size = dtype(0).nbytes
    MB_elems = int(1024**2 / dtype_size)
    MB_target = 64  # target for largest tensor
    N = MB_target * MB_elems
    W_target = int(np.log2(N))
    print(f"== Testing memory leak with {N} elements and {MB_target} MB array ==")
    c = CUSZXCompressor(r2r_error=1e-2, r2r_threshold=1e-2)
    c = qtensor.compression.ProfileCompressor(c)
    _nvsmi_handle = _init_nvsmi()

    As, Bs = W_target - 4, W_target - 2
    common_num = int((As + Bs - W_target) / 2)
    print(f"Common indices: {common_num}, W_target: {W_target}")
    avars = [Var(i) for i in range(As)]
    bvars = [Var(i) for i in range(common_num)] + [
        Var(i) for i in range(As, As + Bs - common_num)
    ]
    print("A vars", avars)
    print("B vars", bvars)
    TA = Tensor.empty("A", avars)
    TA.data = np.random.rand(*TA.shape).astype(dtype)
    TB = Tensor.empty("B", bvars)
    TB.data = np.random.rand(*TB.shape).astype(dtype)

    _mem_histories = []
    for j in range(100):
        res = compressed_contract(
            TA,
            TB,
            avars[:common_num],
            W_target - 1,
            c,
            einsum=cupy.einsum,
            move_data=cupy.array,
        )
        [c.free_compressed(x) for x in res.data]
        print(f"Result indices: {res.indices}")
        print(f"Result: {res}")
        _mem = _get_nvsmi_mem(_nvsmi_handle) / 1024**3
        print(f"== [{j}] Memory usage: {_mem} GB ==")
        _mem_histories.append(_mem)
        print(
            f"== [{j}] Memory history: {[np.round(x, 2) for x in _mem_histories]} GB =="
        )
