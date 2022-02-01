# context: https://github.com/cupy/cupy/issues/5075#issuecomment-820838241

import cupy
import cupyx
import torch

N = 4093
x = cupy.random.rand(N, N, dtype=cupy.float32)
y = cupy.random.rand(N, N, dtype=cupy.float32)

cupy.cuda.device.get_cublas_handle()
print('Cupy:')
print(cupyx.time.repeat(cupy.matmul, (x, y), n_repeat=30))

x = torch.zeros((N, N), dtype=torch.float32, device='cuda:0')
y = torch.zeros((N, N), dtype=torch.float32, device='cuda:0')
print('Torch:')

print(cupyx.time.repeat(torch.matmul, (x, y), n_repeat=30))
