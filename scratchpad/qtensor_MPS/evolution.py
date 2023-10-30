import tensornetwork as tn
import numpy as np
from mps import MPS
from gates import *
import matplotlib.pyplot as plt

# Take cutoff into consideration

cutoff = 1e-8
N = 10
mps = MPS("q", N+1, 2)
evolution_range = np.linspace(0, 1, 5)
js = np.arange(0, N)

magnetization = []
for t in evolution_range:
    mag_j = []

    for j in range(0, N):
        mps.apply_two_qubit_gate(isingHamiltonian(t), [j, j+1])

    for j in range(N, 0):
        mps.apply_two_qubit_gate(isingHamiltonian(t), [j-1, j])

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



