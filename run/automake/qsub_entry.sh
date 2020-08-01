#!/bin/bash
echo "Dear God, I am $(hostname)!"
lscpu
python -c "import matplotlib.pyplot as plt; plt.plot(range(55)); plt.savefig('results/image.png')"
sleep 1
ls results
echo 'ls'
ls
echo 'pwd'
pwd
echo 'test'>test.txt
echo 'ls'
ls
python -c "print('Hello from python')"
