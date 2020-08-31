import tcontract
import numpy as np

arr = np.random.randn(2)
arr = np.array(arr, dtype=np.double)
an = tcontract.example(arr, arr, arr)
print(arr)

arr = np.random.randn(4,8,16) + 1j*np.random.randn(4,8,16)
print(np.sum(arr))

# arbitrary step size dx = 1., y=0.5, dz = 0.25
ans = tcontract.integrate3(arr, 1.0, 1.0, 1.0)
print(ans)
