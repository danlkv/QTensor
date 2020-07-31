# Qensor

## Installation

```bash
git clone --recurse-submodules https://github.com/DaniloZZZ/Qensor
cd Qensor
cd qtree && pip install .
pip install .
```


## Usage

```python
from qensor import QAOA_energy

G = nx.random_regular_graph(3, 10)
gamma, beta = [np.pi/3], [np.pi/2]

E = QAOA_energy(G, gamma, beta)
```

## Get treewidth

```python
from qensor.optimisation.Optimizer import OrderingOptimizer
from qensor.optimisation.TensorNet import QtreeTensorNet
from qensor import QtreeQAOAComposer

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
from qensor.optimisation.Optimizer import TamakiOptimizer
from qensor.optimisation.TensorNet import QtreeTensorNet
from qensor import QtreeQAOAComposer

composer = QtreeQAOAComposer(
	graph=G, gamma=gamma, beta=beta)
composer.ansatz_state()


tn = QtreeTensorNet.from_qtree_gates(composer.circuit)

opt = TamakiOptimizer(wait_time=15) # time in seconds for heuristic algorithm
peo, tn = opt.optimize(tn)
treewidth = opt.treewidth

```
