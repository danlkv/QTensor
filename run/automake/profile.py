import matplotlib.pyplot as plt

import numpy as np

print('Profiling')
print('Job1', 1)

# Performance plot
plt.plot(np.sin(np.linspace(0,5)))

plt.savefig('./results/figure.png')
