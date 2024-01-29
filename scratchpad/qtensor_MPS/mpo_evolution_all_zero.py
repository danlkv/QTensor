import tensornetwork as tn
import numpy as np
from mps import MPS
from mpo import MPOLayer
from gates import *
import matplotlib.pyplot as plt
import argparse
import gc

parser = argparse.ArgumentParser()
parser.add_argument("-N", dest="N", default=5, help="number of qubits", type=int)
parser.add_argument(
    "-rangeend", dest="rangeend", default=1, help="End of range", type=int
)
parser.add_argument("-steps", dest="steps", default=10, help="End of range", type=int)

args = parser.parse_args()

cutoff = 1e-8
N = args.N
range_end = args.rangeend
steps = args.steps

mps = MPS("q", N + 1, 2)
mps.apply_single_qubit_gate(xgate(), 0)
assert mps.get_norm() == 1

mpo = MPOLayer("q", N + 1, 2)
evolution_range = np.linspace(0, range_end, steps)
js = np.array(range(N))
dt = evolution_range[1] - evolution_range[0]

magnetization = []
mag_j = []


for j in js:
    print("for : ", j)
    mpo = MPOLayer("q", N + 1, 2)
    mag_j = []
    mpo.add_single_qubit_gate(zgate(), j)
    mag_j += [mpo.mpo_mps_inner_prod(mps)]
    for t in evolution_range[:-1]:
        print("for t : ", t)
        # for j1 in js:
        #     mpo.add_two_qubit_gate(sigmaRzz(dt), [j1, j1 + 1])
        #     mpo.add_two_qubit_gate(sigmaRzz(dt), [j1, j1 + 1], True)

        for sitej in js:
            mpo.add_single_qubit_gate(sigmaRx(dt), sitej)
            mpo.add_single_qubit_gate(sigmaRx(dt), sitej, True)
        # for j in js:
        #     mpo.add_two_qubit_gate(sigmaRzz(dt), [N - 1 - j, N - j])
        #     mpo.add_two_qubit_gate(sigmaRzz(dt), [N - 1 - j, N - j], True)

        # for sitej in js:
        #     mpo.add_single_qubit_gate(sigmaRx(dt), N - 1 - sitej)
        #     mpo.add_single_qubit_gate(sigmaRx(dt), N - 1 - sitej, True)
        mag_j += [mpo.mpo_mps_inner_prod(mps, True)]
        gc.collect()
    magnetization += [mag_j]

plt.style.context("default")
plt.figure(figsize=(12, 7))

# plot final magnetization
plt.pcolormesh(js, evolution_range, np.real(magnetization).T)
plt.set_cmap("RdYlBu")
plt.colorbar()
plt.title("MPO: Total Z-Magnetization Evolution")
plt.xlabel("Site")
plt.ylabel("time [ $Jt$ ]")

plt.savefig("testmpo_only_single_qubit_gates.png")

# Single qubit evolving, compare with mps evolution
# Single qubit rabii evolution
