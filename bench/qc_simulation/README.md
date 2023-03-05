## Advanced usage

It is possible to glob over inputs and vectorize over outputs
The globbing is possible over remote files

```
main.py process \
    gh://example.com/data/*/*.element \
    results/{X}/{in_file}_y{y}.r \
    -X=1,2 --Y=foo,bar
```

The parent directory for each out file will be created automatically

## Examples

### Easily manage simulation and estimation results

After running preprocess, one can estimate runtime and compare that to actual time to simulate
```bash
# Assume 1GFlop (low-end cpu number)
./main.py estimate preprocess/bris/bris_\*.txt_oalgogreedy.circ estimations/bris/cpu --sim qtensor -M 27 -F 1e9
./main.py estimate preprocess/bris/bris_\*.txt_oalgorgreedy.circ estimations/bris/cpu --sim qtensor -M 27 -F 1e9

rm  -r simulations/bris/*
# Simulate Greedy
./main.py simulate preprocess/bris/bris_\*.txt_oalgogreedy.circ simulations/bris --sim qtensor -M 27
# Simulate RGreedy
./main.py simulate preprocess/bris/bris_\*.txt_oalgorgreedy.circ simulations/bris --sim qtensor -M 27
cat simulations/bris/*rgreedy*
cat estimations/bris/cpu/*rgreedy*
cat simulations/bris/*greedy*
cat estimations/bris/cpu/*greedy*
```

This shows how UNIX utilities are used to filter and present data. In SQL this would be something like
`SELECT * FROM simulations WHERE ordering_algo="greedy"`. 

## Filetypes

- `.txt` - gate sequence as in GRCS
- `.qasm` - openqasm file
- `.jsonterms` - json file of QAOA terms (`src/circuit_gen/qaoa.py`)
