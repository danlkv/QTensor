seq 23 25 | xargs -L1 python transposes.py | grep duration --line-buffered | cut -d'=' -f2
