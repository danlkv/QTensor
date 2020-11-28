## Performance comparison between qtensor and others




```
n=100 p=1 d=3 n_edges =150 n_processes=56 lib=qtensor time=0.23
n=100 p=1 d=3 n_edges =150 n_processes=56 lib=quimb   time=2.7    hyperopt_time=0.01

n=100 p=2 d=3 n_edges =150 n_processes=56 lib=qtensor time=0.34
n=100 p=2 d=3 n_edges =150 n_processes=56 lib=quimb   time=1.8    hyperopt_time=0.01
n=100 p=2 d=3 n_edges =150 n_processes=56 lib=acqdp   time=26

n=100 p=3 d=3 n_edges =150 n_processes=56 lib=qtensor time=0.49
n=100 p=3 d=3 n_edges =150 n_processes=56 lib=quimb   time=2.8    hyperopt_time=0.01

n=100 p=4 d=3 n_edges =150 n_processes=56 lib=qtensor time=45.8
n=100 p=4 d=3 n_edges =150 n_processes=56 lib=quimb   time=48.3   hyperopt_time=10

n=54 p=4 d=3 n_edges=150 n_processes=56 lib=qtensor time=16
n=54 p=4 d=3 n_edges=150 n_processes=56 lib=quimb  time=42        hyperopt_time=10
n=54 p=4 d=3 n_edges=150 n_processes=56 lib=quimb  time=N/A       hyperopt_time=0.01

n=500 p=4 d=3 n_edges=750 n_processes=56 lib=qtensor time=5.59
n=500 p=4 d=3 n_edges=750 n_processes=56 lib=quimb   time=196.    hyperopt_time=10

n=1000 p=4 d=3 n_edges=1500 n_processes=56 lib=qtensor time=11.11
n=1000 p=4 d=3 n_edges=1500 n_processes=56 lib=quimb   time=>300   hyperopt_time=10

n=1000 p=3 d=3 n_edges=1500 n_processes=56 lib=qtensor time=4.89
n=1000 p=3 d=3 n_edges=1500 n_processes=56 lib=quimb   time=101     hyperopt_time=1
n=1000 p=3 d=3 n_edges=1500 n_processes=56 lib=quimb   time=96      hyperopt_time=0.3
n=1000 p=3 d=3 n_edges=1500 n_processes=56 lib=acqdp   time=N/A

n=1000 p=2 d=3 n_edges=1500 n_processes=56 lib=qtensor time=1.54
n=1000 p=2 d=3 n_edges=1500 n_processes=56 lib=quimb   time=44      hyperopt_time=.3
n=1000 p=2 d=3 n_edges=1500 n_processes=56 lib=quimb   time=31      hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=56 lib=acqdp   time=260

```

## Dependence on seed and machine

```

n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=10 lib=quimb   time=35.05      hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=10 lib=quimb   time=37.51      hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=10 lib=quimb   time=38.18      hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=10 lib=quimb   time=35.63      hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=28 seed=10 lib=quimb   time=60.14      hyperopt_time=.01

n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=11 lib=quimb   time=33.96      hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=11 lib=quimb   time=34.01      hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=11 lib=quimb   time=33.93      hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=11 lib=quimb   time=34.85      hyperopt_time=.01

n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=12 lib=quimb   time=33.92      hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=12 lib=quimb   time=34.03      hyperopt_time=.01

n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=13 lib=quimb   time=34.45      hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=13 lib=quimb   time=35.65      hyperopt_time=.01

n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=14 lib=quimb   time=35.08      hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=14 lib=quimb   time=37.65      hyperopt_time=.01

n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=10 lib=qtensor   time=1.457 
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=10 lib=qtensor   time=1.457 

n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=11 lib=qtensor   time=1.420 
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=11 lib=qtensor   time=1.650 
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=11 lib=qtensor   time=1.330 

n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=12 lib=qtensor   time=1.393 
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=12 lib=qtensor   time=1.444 

n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=13 lib=qtensor   time=1.479
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=13 lib=qtensor   time=1.649 

n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=14 lib=qtensor   time=1.450
n=1000 p=2 d=3 n_edges=1500 n_processes=56 seed=14 lib=qtensor   time=1.440 

n=1000 p=4 d=3 n_edges=1500 n_processes=56 seed=14 lib=qtensor   time=11.5
n=1000 p=4 d=3 n_edges=1500 n_processes=56 seed=14 lib=qtensor   time=11.3
n=1000 p=4 d=3 n_edges=1500 n_processes=56 seed=14 lib=qtensor   time=11.6

n=1000 p=4 d=3 n_edges=1500 n_processes=56 seed=12 lib=qtensor   time=11.05
n=1000 p=4 d=3 n_edges=1500 n_processes=56 seed=12 lib=qtensor   time=10.97

n=1000 p=4 d=3 n_edges=1500 n_processes=56 seed=11 lib=qtensor   time=11.13
n=1000 p=4 d=3 n_edges=1500 n_processes=56 seed=13 lib=qtensor   time=11.19
n=1000 p=4 d=3 n_edges=1500 n_processes=56 seed=10 lib=qtensor   time=11.31
```
