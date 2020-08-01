#!/bin/bash
echo "Dear God, I am $(hostname)!"
lscpu
python -c "import matplotlib.pyplot as plt; plt.plot(range(55)); plt.savefig('results/image.png')"
ls
pwd
python -c "print('Hello from python')"
