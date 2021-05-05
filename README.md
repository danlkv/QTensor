

## Installation

```bash
pip install qtensor # currently has a bug, do not use
```
Or from source:
```bash
git clone --recurse-submodules https://github.com/DaniloZZZ/QTensor
cd QTensor
cd qtree && pip install . && cd ..
pip install .
```

### Docker image

https://hub.docker.com/repository/docker/danlkv/qtensor

## Usage

```python
from qtensor import QAOA_energy

G = nx.random_regular_graph(3, 10)
gamma, beta = [np.pi/3], [np.pi/2]

E = QAOA_energy(G, gamma, beta)
```

## Get treewidth

```python
from qtensor.optimisation.Optimizer import OrderingOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor import QtreeQAOAComposer

composer = QtreeQAOAComposer(
	graph=G, gamma=gamma, beta=beta)
composer.ansatz_state()


tn = QtreeTensorNet.from_qtree_gates(composer.circuit)

opt = OrderingOptimizer()
peo, tn = opt.optimize(tn)
treewidth = opt.treewidth

```

### Tamaki solver

#### Instalation

The tamaki solver repository should be already cloned into
`QTensor/qtree/thirdparty/tamaki_treewidth`.

To compile it, go to the directory and run `make heuristic`.

```bash
> cd QTensor/qtree/thirdparty/tamaki_treewidth
> make heuristic 
javac tw/heuristic/*.java
```

Tamaki solver repository: https://github.com/TCS-Meiji/PACE2017-TrackA


If you have memory errors, modify the `JFLAGS` variable in the bash script `./tw-heuristic`. I use `JFLAGS="-Xmx4g -Xms4g -Xss500m"`.

#### Usage

```python
from qtensor.optimisation.Optimizer import TamakiOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor import QtreeQAOAComposer

composer = QtreeQAOAComposer(
	graph=G, gamma=gamma, beta=beta)
composer.ansatz_state()


tn = QtreeTensorNet.from_qtree_gates(composer.circuit)

opt = TamakiOptimizer(wait_time=15) # time in seconds for heuristic algorithm
peo, tn = opt.optimize(tn)
treewidth = opt.treewidth

```
#### Use tamaki for QAOA energy

and also raise an error when treewidth is large.

```python
from qtensor.optimisation.Optimizer import TamakiOptimizer
from qtensor import QAOAQtreeSimulator

class TamakiQAOASimulator(QAOAQtreeSimulator):
    def optimize_buckets(self):
        opt = TamakiOptimizer()
        peo, self.tn = opt.optimize(self.tn)
        if opt.treewidth > 30:
            raise Exception('Treewidth is too large!')
        return peo

sim = TamakiQAOASimulator(QtreeQAOAComposer)

if n_processes:
    res = sim.energy_expectation_parallel(G, gamma=gamma, beta=beta
        ,n_processes=n_processes
    )
else:
    res = sim.energy_expectation(G, gamma=gamma, beta=beta)
return res

```

### Useful features

- raise ValueError if treewidth is too large:
```python
sim = QAOAQtreeSimulator(max_tw=24)
sim.energy_expectation(G, gamma=gamma, beta=beta)
```

- generate graphs

```python
from qtree.toolbox import random_graph

G_reg = random_graph(12, type='random', degree=3, seed=42)
G_er = random_graph(12, type='erdos_renyi', degree=3, seed=42)

```
- get cost estimation

```python
from qtensor.optimisation.Optimizer import TamakiOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor import QtreeQAOAComposer

composer = QtreeQAOAComposer(
	graph=G, gamma=gamma, beta=beta)
composer.ansatz_state()

tn = QtreeTensorNet.from_qtree_gates(composer.circuit)

opt = TamakiOptimizer(wait_time=15)
peo, tn = opt.optimize(tn)
treewidth = opt.treewidth
mems, flops = tn.simulation_cost(peo)
print('Max memory=', max(mems), 'Total flops=', sum(flops))
```
- get QAOA cost estimation

```python
from qtensor.toolbox import qaoa_energy_cost_params_from_graph

costs_per_edge = qaoa_energy_cost_params_from_graph(graph, p,
        ordering_algo='greedy', max_time=60)

tws, mems, flops = zip(*costs_per_edge)
print('Max treewidth=', max(tws), 'Max memory=', max(mems), 'Total flops=', sum(flops))
```
