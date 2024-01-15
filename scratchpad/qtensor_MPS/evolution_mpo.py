import tensornetwork as tn
import numpy as np
from mps import MPS
from mpo import MPOLayer
from gates import *
import matplotlib.pyplot as plt

## See the mpo prepared and at the end the zgate thing
## Compare resuts
cutoff = 1e-8
N = 5
mps = MPS("q", N + 1, 2)
mps.apply_single_qubit_gate(xgate(), 0)
# mps.apply_single_qubit_gate(xgate(), 4)
assert mps.get_norm() == 1
# print(mps.get_wavefunction())

mpo = MPOLayer("q", N + 1, 2)
evolution_range = np.linspace(0, 5, 30)
js = np.array(range(N))

magnetization = []

mag_j = []

# for j in range(0, N):
#     mpo.add_single_qubit_gate(zgate(), j)
#     mag_j += [mpo.mpo_mps_inner_prod(mps)]
#     mpo.add_single_qubit_gate(zgate(), j)

# magnetization += [mag_j]

dt = 25 / 20

for j in range(0, N):
    mpo = MPOLayer("q", N + 1, 2)
    mpo.add_single_qubit_gate(zgate(), j)
    mag_j = []
    mag_j += [mpo.mpo_mps_inner_prod(mps)]
    for t in evolution_range:
        # for j in js:
        # mpo.add_two_qubit_gate(sigmaRzz(dt), [j, j + 1])
        # mpo.add_two_qubit_gate(sigmaRzz(dt), [j, j + 1], True)

        for j in js:
            mpo.add_single_qubit_gate(sigmaRx(dt), j)
            mpo.add_single_qubit_gate(sigmaRx(dt), j, True)

        # for j in js:
        #     mpo.add_two_qubit_gate(sigmaRzz(-1 * t), [N - 1 - j, N - j])
        #     mpo.add_two_qubit_gate(sigmaRzz(-1 * t), [N - 1 - j, N - j], True)

        # for j in js:
        #     mpo.add_single_qubit_gate(sigmaRx(-1 * t), N - 1 - j)
        #     mpo.add_single_qubit_gate(sigmaRx(-1 * t), N - 1 - j, True)

        # mps.apply_mpo_layer(mpo)
        # ZIIIII..
        # IZIIII...
        # IIZIII...

        mag_j += [mpo.mpo_mps_inner_prod(mps)]

    magnetization += [mag_j]

plt.style.context("default")
plt.figure(figsize=(12, 7))

# plot final magnetization
plt.pcolormesh(js, np.linspace(0, 5, 31), np.real(magnetization).T)
plt.set_cmap("RdYlBu")
plt.colorbar()
plt.title("Total Z-Magnetization Evolution")
plt.xlabel("Site")
plt.ylabel("time [ $Jt$ ]")

plt.savefig("mpo_final_magnetisation.png")

# Single qubit evolving, compare with mps evolution
# Single qubit rabii evolution
