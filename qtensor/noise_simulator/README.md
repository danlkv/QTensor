# Noisy simulations using QTensor

This folder uses stochastic simulation of quantum circuits
and benchmarks it against qiskit density matrix simulations.


## Observations

Two types of circuits are considered:

1. QAOA MaxCut with CZ two-qubit gate
2. Randomized brickwork circuit with XYZ gates and CX gate

The noise model is depolarizing noise applied to every quantum gate.

Two types of qiskit runs were considered: with passing noise_model to `execute` and to `backend`. These modes are named E and B below.
Mode B is the correct one.

The qiskit simulation results are then compared to qtensor simulation results and the discrepancy between the two is evaluated.

### 1. QAOA circuits

When running with mode E, we see that there is very small error and it reduces with number of circuits k. The same was with mode B.

### 2. Generic circuits

When running in mode E, we found that the error does not converge with number of circuits K. 
In the mode B we found that the error is larger but converges better (by how much?) K, which indicates that we compare against the correct noisy output.

When running in mode E, removing two-qubit gate error channels reduced the simulation error.

When running in mode E, adding a layer of Hadamards in the begining of the circuit reduced the error significantly.

When running in both modes E and B, increasing error rate in channel reduces the simulation error.

In some cases hovewer, there was convergence of error in mode E

These results are preliminary.
