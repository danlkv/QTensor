# QTensor

## Installation

```bash
git clone --recurse-submodules https://github.com/DaniloZZZ/QTensor
cd QTensor
cd qtree && pip install .
pip install .
```


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

### Use tamaki solver


Install from here: https://github.com/TCS-Meiji/PACE2017-TrackA

If you have memory errors, modify the `JFLAGS` variable in bash script `./tw-heuristic`. I use `JFLAGS="-Xmx4g -Xms4g -Xss500m"`.

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

class TamakiQAOASimulator(QAOAQtreeSimsulator):
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


### Use cli to run benchmarks

```bash
» python -m qensor.cli generate-qaoa-ansatz-circuit -p 3 -n 24 | python -m qensor.cli sim-file --profile --target-tw 27
```
