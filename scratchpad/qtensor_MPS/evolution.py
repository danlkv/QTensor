import tensornetwork as tn
import numpy as np
from mps import MPS
from gates import *
import matplotlib.pyplot as plt

N = 10
cutoff = 1e-8
tau = 0.1
ttotal = 2.0

gates = []
N = 25
n = int(N/2)
mps = MPS("q", N, 2)
evolution_range = np.linspace(0, 1, 5)
js = np.arange(0, n)

magnetization = []
for t in evolution_range:
    mag_t = []
    for j in range(0, n):
        mps.apply_two_qubit_gate(isingHamiltonian(t), [j, j+1])

        mag_t += [mps.get_expectation(zgate(), j)]

    magnetization += [mag_t]

with plt.style.context('default'):
    plt.figure(figsize=(12, 7))

    # plot the magnetization
    ax1 = plt.subplot(131)
    plt.pcolormesh(js, evolution_range, np.real(magnetization))
    plt.set_cmap('RdYlBu')
    plt.colorbar()
    plt.title('Z-Magnetization')
    plt.xlabel('Site')
    plt.ylabel('time [ $Jt$ ]')

    plt.show()



