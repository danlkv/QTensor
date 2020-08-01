#!/bin/bash
echo "Dear God, I am $(hostname)!"
lscpu
echo $PYTHONPATH
echo $PATH
echo $SHELL
python -c "import matplotlib.pyplot as plt; plt.plot(range(55)); plt.savefig('results/image.png')"
echo 'pwd'
pwd
echo 'test'>test.txt
echo 'ls'
ls
python -c "print('Hello from python')"
