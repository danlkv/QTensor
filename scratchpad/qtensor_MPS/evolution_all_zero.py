import tensornetwork as tn
import numpy as np
from mps import MPS
from gates import *
import matplotlib.pyplot as plt

# TODO : Take cutoff into consideration
# for magnetization

# TODO: Check cutoff and tolerance

# Check state 0 and then check expectation 1 0 0 0
# Check singular values 
# Check sum of prob going > 1

# Plot initial values at t = 0

cutoff = 1e-8
N = 10
mps = MPS("q", N+1, 2)
evolution_range = np.linspace(0, 1, 20)
js = np.array(range(N))

# converting |0000..0..00> to |0000..1..00>
# mps.apply_single_qubit_gate(xgate(), 2)

assert(mps.get_norm() == 1)
print(mps.get_wavefunction())
# magnetization = []

# # TODO:
# # Try for 10-15 qubits
# # TEST: Smooth evolution for smaller time steps
# # TEST: At delta t = 0 no changes

# # Initial expectation values
# for j in range(0, N):
#     magnetization.append(mps.get_expectation(zgate(), j))

# print(magnetization)
# plt.style.context('default')
# plt.figure(figsize=(12, 7))

# # plot initial magnetization
# X = [x.real for x in magnetization]
# Y = [x.imag for x in magnetization]

# plt.plot(X, 'ro')
# plt.ylabel('Imaginary') 
# plt.xlabel('Real')
# plt.savefig('initial_magnetisation.png')

# TODO: 
# 1. Implement matrix exponentatiation
# 2. Initialise the magnestiziation and plot the in colormesh
# 3. Then run the second part of code.

# # reset magnetization
magnetization = []

for t in evolution_range:
    mag_j = []

    for j in js:
        mps.apply_two_qubit_gate(sigmaRZZ(t), [j, j+1])

    for j in js:
        mps.apply_two_qubit_gate(sigmaRZZ(t), [N-1-j, N-j])
    
    print(mps.get_wavefunction())
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