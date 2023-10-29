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
N = 10
n = 3
mps = MPS("q", N, 2)
evolution_range = np.linspace(0, 80, 7)
js = np.arange(0, n)

magnetization = []

for t in evolution_range:
    mag_t = []
    for j in range(0, n):
        gate = sigmaZZ(t)
        print(gate.tensor , t, j)
        mps.apply_two_qubit_gate(sigmaZZ(t), [j, j+1])
        mps.apply_two_qubit_gate(sigmaXnegXpos(t), [j, j+1])
        mps.apply_two_qubit_gate(sigmaXposXneg(t), [j, j+1])

        mag_t += [mps.get_expectation(zgate(), j)]

    magnetization += [mag_t]

with plt.style.context('default'):
    plt.figure(figsize=(12, 7))

    # plot the magnetization
    ax1 = plt.subplot(131)
    plt.pcolormesh(js, evolution_range, np.real(magnetization), vmin=-0.5, vmax=0.5)
    plt.set_cmap('RdYlBu')
    plt.colorbar()
    plt.title('Z-Magnetization')
    plt.xlabel('Site')
    plt.ylabel('time [ $Jt$ ]')

    plt.show()



