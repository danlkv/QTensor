import tensornetwork as tn
import numpy as np
from mps import MPS
from gates import *
import matplotlib.pyplot as plt

cutoff = 1e-8
N = 3
mps = MPS("q", N+1, 2)
evolution_range = np.linspace(0, 1, 100)
js = np.array(range(N))

magnetization = []

for t in evolution_range:
    mag_j = []

    for j in js:
        mps.apply_two_qubit_gate(sigmaRZZ(t), [j, j+1])

    for j in js:
        mps.apply_two_qubit_gate(sigmaRZZ(t), [N-1-j, N-j])

    # ZIIIII..
    # IZIIII...
    # IIZIII...
    for j in range(0, N):
        mag_j += [mps.get_expectation(zgate(), j)]

    magnetization += [mag_j]

plt.style.context('default')
plt.figure(figsize=(12, 7))

# plot final magnetization
plt.pcolormesh(js, evolution_range, np.real(magnetization))
plt.set_cmap('RdYlBu')
plt.colorbar()
plt.title('Total Z-Magnetization Evolution')
plt.xlabel('Site')
plt.ylabel('time [ $Jt$ ]')

plt.savefig('final_magnetisation_all0.png')