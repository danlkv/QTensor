
## Examples

1. generate or download circuits:

* As tar `./main.py echo github://danlkv:GRCS@/inst/bristlecone/cz_v2/bris_11.tar.gz data/circuits/bris11/\{in_file\}.circ` (need to unzip)
* Using http and [unzip on the fly](./scripts/http_unzip_on_the_fly.sh)
* generate `./main.py generate data/circuits/qaoa/maxcut_regular_N{N}_p{p} --type=qaoa_maxcut --N=8,12,16,24,32,48,64 --p=1,2,3,4,5 --d=3`

2. preprocess using both of `greedy` and `rgreedy` algorithms:
`./main.py preprocess data/circuits/qaoa/maxcut_regular\* data/preprocess/maxcut/\{in_file\}_oalgo{O}.circ --O=greedy,rgreedy --sim=qtensor
`
3. Simulate: `./main.py simulate ./data/preprocess/maxcut/maxcut_regular\* data/simulations/maxcut/{in_file}_comp_m{M} --sim qtensor -M 25 --backend=cupy --compress=szx`

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
