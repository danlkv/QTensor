# Qensor

## Installation

```bash
git clone ----recurse-submodules https://github.com/DaniloZZZ/Qensor
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
