import tensornetwork as tn
import numpy as np
from mps import MPS
from mpo import MPOLayer
from gates import *
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-N", dest="N", default=2, help="number of qubits", type=int)
parser.add_argument(
    "-rangeend", dest="rangeend", default=2, help="End of range", type=float
)
parser.add_argument("-steps", dest="steps", default=10, help="End of range", type=int)

args = parser.parse_args()

cutoff = 1e-8
N = args.N
range_end = args.rangeend
steps = args.steps
mps = MPS("q", N + 1, 2)
assert mps.get_norm() == 1

mps.apply_single_qubit_gate(xgate(), 0)
mps.apply_single_qubit_gate(xgate(), N - 1)

mpo = MPOLayer("q", N + 1, 2)
evolution_range = np.linspace(0, range_end, steps)
js = np.array(range(N))
dt = 0 if len(evolution_range) <= 1 else evolution_range[1] - evolution_range[0]

magnetization = []
mag_j = []


for j in js:
    mpo = MPOLayer("q", N + 1, 2)
    mag_j = []
    mpo.add_single_qubit_gate(zgate(), j)

    for i in range(N):
        print(mpo._nodes[i].tensor)

    print("\n\n")

    mag_j += [mpo.mpo_mps_inner_prod(mps)]
    print("mag initially: ", mag_j, " at j: ", j)
    for t in evolution_range[:-1]:
        print("At t = ", t)
        # Two qubit gate
        for sitej in js[:-1]:
            mpo.add_two_qubit_gate(sigmaRzz(dt), [sitej, sitej + 1])
            print(
                "mpo after applying sigmaRzz at site: \n",
                N - 1 - sitej,
                "and",
                N - sitej,
                "\n",
                mpo._nodes[sitej].tensor,
                mpo._nodes[sitej + 1].tensor,
                "\n\n",
            )
            mpo.add_two_qubit_gate(sigmaRzz(dt), [sitej, sitej + 1], True)
            print(
                "mpo after applying conj sigmaRzz at site: \n",
                N - 1 - sitej,
                "and",
                N - sitej,
                "\n",
                mpo._nodes[sitej].tensor,
                mpo._nodes[sitej + 1].tensor,
                "\n\n",
            )

        # Single qubit gate
        for sitej in js:
            mpo.add_single_qubit_gate(sigmaRx(dt), sitej)
            print(
                "mpo after applying sigmaRx : at site: ",
                sitej,
                "\n",
                mpo._nodes[sitej].tensor,
                "\n\n",
            )
            mpo.add_single_qubit_gate(sigmaRx(dt), sitej, True)
            print(
                "mpo after applying conj sigmaRx : at site:",
                sitej,
                "\n",
                mpo._nodes[sitej].tensor,
                " \n\n",
            )

        # # Two qubit gate
        for sitej in js[1:]:
            mpo.add_two_qubit_gate(sigmaRzz(dt), [N - 1 - sitej, N - sitej])
            print(
                "mpo after applying sigmaRzz at site: \n",
                N - 1 - sitej,
                "and",
                N - sitej,
                "\n",
                mpo._nodes[N - 1 - sitej].tensor,
                mpo._nodes[N - sitej].tensor,
                "\n\n",
            )
            mpo.add_two_qubit_gate(sigmaRzz(dt), [N - 1 - sitej, N - sitej], True)
            print(
                "mpo after applying conj sigmaRzz at site: \n",
                N - 1 - sitej,
                "and",
                N - sitej,
                mpo._nodes[N - 1 - sitej].tensor,
                mpo._nodes[N - sitej].tensor,
                "\n\n",
            )

        # Single qubit gate
        for sitej in js:
            mpo.add_single_qubit_gate(sigmaRx(dt), N - 1 - sitej)
            print(
                "mpo after applying sigmaRx at site: ",
                N - 1 - sitej,
                "\n",
                mpo._nodes[sitej].tensor,
                "\n\n",
            )
            mpo.add_single_qubit_gate(sigmaRx(dt), N - 1 - sitej, True)
            print(
                "mpo after applying conj sigmaRx at site:",
                N - 1 - sitej,
                "\n",
                mpo._nodes[sitej].tensor,
                "\n\n",
            )

        mag_j += [mpo.mpo_mps_inner_prod(mps)]
        print("mag after mpo evolution: ", mag_j, " at t: ", t, " \n")

    magnetization += [mag_j]


print(f"Total magnetization is: {np.real(magnetization).T}, \n")
print("--------------------------------------------------\n")

plt.style.context("default")
plt.figure(figsize=(12, 7))

# plot final magnetization
plt.pcolormesh(js, evolution_range, np.real(magnetization).T)
plt.set_cmap("RdYlBu")
plt.colorbar()
plt.title("MPO: Total Z-Magnetization Evolution")
plt.xlabel("Site")
plt.ylabel("time [ $Jt$ ]")

plt.savefig("test_mpo_evolution.png")

# Single qubit evolving, compare with mps evolution
# Single qubit rabii evolution
