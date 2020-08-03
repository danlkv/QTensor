#!/bin/zsh
echo "Dear God, I am $(hostname)!"

[[ -e ~/.profile ]] && emulate sh -c 'source ~/.profile'
source ~/.profile
export PATH="/home/danlkv/.local/bin:/home/danlkv/bin:$PATH"
export PYTHONPATH="/home/danlkv/git-build/cpython/Lib:$PYTHONPATH"
#                                                                                   
lscpu
echo $PYTHONPATH
echo $PATH
echo $SHELL
python ./profile.py > results/profile.txt
python -c "import matplotlib.pyplot as plt; plt.plot(range(55)); plt.savefig('results/image.png')"
echo 'test'>test.txt
python -c "print('Hello from python')"
