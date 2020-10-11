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
qtensor-specs-time-flop-plot time-vs-flops-plot results/time_vs_flops.png --backend=mkl --max-memory=5e10 --min-memory=1e6 --seed=111 --ordering_algo=tamaki_10 > results/time_vs_flops.txt
