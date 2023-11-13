import tensornetwork as tn
import numpy as np
from mps import MPS
from gates import *
import matplotlib.pyplot as plt

cutoff = 1e-8
N = 10
mps = MPS("q", N + 1, 2)
evolution_range = np.linspace(0, 1, 10)
js = np.array(range(N))


# converting |0000..0..00> to |1001001001>
mps.apply_single_qubit_gate(xgate(), 0)
mps.apply_single_qubit_gate(xgate(), 3)
mps.apply_single_qubit_gate(xgate(), 6)
mps.apply_single_qubit_gate(xgate(), 9)

assert mps.get_norm() == 1
print(mps.get_wavefunction())

# TODO:
# 1. Implement matrix exponentatiation
# 2. Initialise the magnestiziation and plot the in colormesh
# 3. Then run the second part of code.

# # reset magnetization
magnetization = []

for t in evolution_range:
    mag_j = []

    for j in js:
        mps.apply_two_qubit_gate(sigmaRzz(t), [j, j + 1])
        mps.apply_single_qubit_gate(sigmaRx(t), j)

    # for j in js:
    #     mps.apply_two_qubit_gate(sigmaRzz(t), [N-1-j, N-j])
    # mps.apply_single_qubit_gate(sigmaRx(t), N-1-j)

    # ZIIIII..
    # IZIIII...
    # IIZIII...
    for j in range(0, N):
        mag_j += [mps.get_expectation(zgate(), j)]

    magnetization += [mag_j]

plt.style.context("default")
plt.figure(figsize=(12, 7))

# plot final magnetization
plt.pcolormesh([str(i) for i in js], evolution_range, np.real(magnetization))
plt.set_cmap("RdYlBu")
plt.colorbar()
plt.title("Total Z-Magnetization Evolution")
plt.xlabel("Site")
plt.ylabel("time [ $Jt$ ]")

plt.savefig("final_magnetisation.png")
