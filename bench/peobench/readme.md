## About
`peo_bench.py` is designed to benchmark various Perfect Elimination Order (peo) solvers. Usage depends on specificying a solver, its options, and a problem graph on which the solver will run. The script is designed to output all data into the console so that it may be piped into some external database.

## How To Run
All options can be modified in the functions themselves. Then the function can run for the same problem using different solvers, graph options, and seeds for the random regular graphs. Uncomment the function to be run in `peo_bench.py` and add your options to the array in the function. 

Note that the function to call for every benchmark is `peo_benchmark_wrapper`. There you can see all the options that need to be entered into the function. 

## Collecting Data
The resulting data is structured to be output into the console and piped into MongoDB using `mongocat`
```
$ export MONGOCAT_URL='mongodb://<username>:<password>@your-mongo-host'
$ python3 peo_bench.py | mongocat -W -d tensim peo_benchmarks
```

## Room for improvement
Data collection is serial. Either some loops should be made to run in parallel or `peo_benchmark_wrapper` should be rewritten as a bash script to run on multiple screens.