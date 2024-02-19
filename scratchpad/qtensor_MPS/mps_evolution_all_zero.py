import tensornetwork as tn
import numpy as np
from mps import MPS
from gates import *
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-N", dest="N", default=5, help="number of qubits", type=int)
parser.add_argument(
    "-rangeend", dest="rangeend", default=2, help="End of range", type=int
)
parser.add_argument("-steps", dest="steps", default=10, help="End of range", type=int)

args = parser.parse_args()

cutoff = 1e-8
N = args.N
range_end = args.rangeend
steps = args.steps
mps = MPS("q", N + 1, 2)
assert mps.get_norm() == 1

# mps.apply_single_qubit_gate(xgate(), 0)
# mps.apply_single_qubit_gate(xgate(), N - 1)

evolution_range = np.linspace(0, range_end, steps)
js = np.array(range(N))
dt = 0 if len(evolution_range) <= 1 else evolution_range[1] - evolution_range[0]
magnetization = []
mag_j = []

for j in range(0, N):
    print(f"tensor at {j} :: {mps._nodes[j].tensor}")

for j in range(0, N):
    mag_j += [mps.get_expectation(zgate(), j)]

magnetization += [mag_j]

# TODO:
# Check for period of rotations as we change dt

for t in evolution_range[:-1]:
    print("At t = ", t)
    mag_j = []

    for j in js[:-1]:
        mps.apply_two_qubit_gate(sigmaRzz(dt), [j, j + 1])
        print(
            f"After apply sigmaRzz tensor :: \n {mps._nodes[0].tensor} \n {mps._nodes[1].tensor}"
        )

    for j in js:
        mps.apply_single_qubit_gate(sigmaRx(dt), j)
        print(f"After applying SigmaRx tensor at {j} now :: {mps._nodes[j].tensor}")

    for j in js[1:]:
        mps.apply_two_qubit_gate(sigmaRzz(dt), [N - 1 - j, N - j])
        print(
            f"After apply sigmaRzz tensor :: \n {mps._nodes[0].tensor} \n {mps._nodes[1].tensor}"
        )

    for j in js:
        mps.apply_single_qubit_gate(sigmaRx(dt), N - 1 - j)
        print(
            f"After applying SigmaRx tensor at {N - 1 - j} now :: {mps._nodes[j].tensor}"
        )

    # ZIIIII..
    # IZIIII...
    # IIZIII...
    for j in range(0, N):
        mag_j += [mps.get_expectation(zgate(), j)]

    print("mag after mps evolution: ", mag_j, " at t: ", t, " \n")
    magnetization += [mag_j]

print(f"Total magnetization is: {np.real(magnetization)}")

plt.style.context("default")
plt.figure(figsize=(12, 7))

print(len(js), len(evolution_range), len(np.real(magnetization)))
# plot final magnetization
plt.pcolormesh(js, evolution_range, np.real(magnetization))
plt.set_cmap("RdYlBu")
plt.colorbar()
plt.title("MPS: Total Z-Magnetization Evolution")
plt.xlabel("Site")
plt.ylabel("time [ $Jt$ ]")

plt.savefig("test_mps_evolution.png")

# o = zgate at j
# MPS = 00000 (s), MPO = u'ou
# inner_prod -> <MPS|u'ou|MPS>, MPS = u'ou |s>,
# #
