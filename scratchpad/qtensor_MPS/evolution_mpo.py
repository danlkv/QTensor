import tensornetwork as tn
import numpy as np
from mps import MPS
from mpo import MPOLayer
from gates import *
import matplotlib.pyplot as plt

## See the mpo prepared and at the end the zgate thing
## Compare resuts
cutoff = 1e-8
N = 10
mps = MPS("q", N + 1, 2)
mps.apply_single_qubit_gate(xgate(), 0)
mps.apply_single_qubit_gate(xgate(), 9)
assert mps.get_norm() == 1
print(mps.get_wavefunction())

mpo = MPOLayer("q", N + 1, 2)
pauli_string = "Z" * (N + 1)
mpo.construct_mpo(pauli_string)
evolution_range = np.linspace(0, 5, 20)
js = np.array(range(N))

magnetization = []

mag_j = []

for j in range(0, N):
    mag_j += [mps.get_expectation(zgate(), j)]

magnetization += [mag_j]

dt = 20 / 20

for t in evolution_range:
    mag_j = []

    for j in js:
        mpo.add_two_qubit_gate(sigmaRzz(dt), [j, j + 1])
        mpo.add_two_qubit_gate(sigmaRzz(dt), [j, j + 1], True)

    for j in js:
        mpo.add_single_qubit_gate(sigmaRx(dt), j)
        mpo.add_single_qubit_gate(sigmaRx(dt), j, True)

    for j in js:
        mpo.add_two_qubit_gate(sigmaRzz(-1 * t), [N - 1 - j, N - j])
        mpo.add_two_qubit_gate(sigmaRzz(-1 * t), [N - 1 - j, N - j], True)

    for j in js:
        mpo.add_single_qubit_gate(sigmaRx(-1 * t), N - 1 - j)
        mpo.add_single_qubit_gate(sigmaRx(-1 * t), N - 1 - j, True)

    # # ZIIIII..
    # # IZIIII...
    # # IIZIII...
    for j in range(0, N):
        mag_j += [mpo.mpo_mps_inner_prod(mps)]

    magnetization += [mag_j]

plt.style.context("default")
plt.figure(figsize=(12, 7))

# plot final magnetization
plt.pcolormesh(js, np.linspace(0, 5, 21), np.real(magnetization))
plt.set_cmap("RdYlBu")
plt.colorbar()
plt.title("Total Z-Magnetization Evolution")
plt.xlabel("Site")
plt.ylabel("time [ $Jt$ ]")

plt.savefig("final_magnetisation_all0_withmpo.png")
