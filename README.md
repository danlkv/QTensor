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
