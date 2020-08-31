import tcontract
import sys
import numpy as np

def test(func):
    def wraped():
        print('Testing', func.__name__)
        func()
    return wraped

@test
def test_transpose():
    arr = np.array([[0,1],[2,3]])
    arr = np.array(arr, dtype=np.double)
    print('in python:\n', arr)
    _ = tcontract.print_4(arr)

    arr = arr.T
    print('in python:\n', arr)
    _ = tcontract.print_4(arr)

test_transpose()

@test
def test_transpose_large():
    N = 25
    arr = np.random.randn(*[2]*N)
    tcontract.print_4(arr)
    print('transposed')
    arr = arr.transpose(*reversed(range(N)))
    tcontract.print_4(arr)

test_transpose_large()

arr = np.random.randn(4,8,16) + 1j*np.random.randn(4,8,16)
print(np.sum(arr))

# arbitrary step size dx = 1., y=0.5, dz = 0.25
ans = tcontract.integrate3(arr, 1.0, 1.0, 1.0)
print(ans)
