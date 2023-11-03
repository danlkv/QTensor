import tensornetwork as tn
import numpy as np
from mps import MPS
from gates import *
import matplotlib.pyplot as plt

# TODO : Take cutoff into consideration
# for magnetization

# TODO: Check cutoff and tolerance

# Check state 0 and then check expectation 1 0 0 0
# Check singular values 
# Check sum of prob going > 1

cutoff = 1e-8
N = 20
mps = MPS("q", N+1, 2)
evolution_range = np.linspace(0, 80, 20)
js = np.array(range(N))

magnetization = []
for t in evolution_range:
    mag_j = []

    for j in js:
        mps.apply_two_qubit_gate(sigmaZZ(t), [j, j+1])

    for j in js:
        mps.apply_two_qubit_gate(sigmaZZ(t), [N-1-j, N-j])

    for j in range(0, N):
        mag_j += [mps.get_expectation(zgate(), j)]

    magnetization += [mag_j]

plt.style.context('default')
plt.figure(figsize=(12, 7))

# plot the magnetization
plt.pcolormesh(js, evolution_range, np.real(magnetization), vmin=-0.5, vmax=0.5)
plt.set_cmap('RdYlBu')
plt.colorbar()
plt.title('Z-Magnetization')
plt.xlabel('Site')
plt.ylabel('time [ $Jt$ ]')

plt.savefig('magnetisation.png')



