## Performance comparison between qtensor and others




```
hyperopt_time=0.01
n=100 p=1 d=3 n_edges =150 n_processes=56 lib=qtensor time=0.23
n=100 p=1 d=3 n_edges =150 n_processes=56 lib=quimb   time=2.7

n=100 p=2 d=3 n_edges =150 n_processes=56 lib=qtensor time=0.34
n=100 p=2 d=3 n_edges =150 n_processes=56 lib=quimb   time=1.8
n=100 p=2 d=3 n_edges =150 n_processes=56 lib=acqdp   time=26

n=100 p=3 d=3 n_edges =150 n_processes=56 lib=qtensor time=0.49
n=100 p=3 d=3 n_edges =150 n_processes=56 lib=quimb   time=2.8

n=100 p=4 d=3 n_edges =150 n_processes=56 lib=qtensor time=45.8
n=100 p=4 d=3 n_edges =150 n_processes=56 lib=quimb   time=48.3 hyperopt_time=10

n=54 p=4 d=3 n_edges=150 n_processes=56 lib=qtensor time=16
n=54 p=4 d=3 n_edges=150 n_processes=56 lib=quimb  time=42     hyperopt_time=10

n=500 p=4 d=3 n_edges=750 n_processes=56 lib=qtensor time=5.59
n=500 p=4 d=3 n_edges=750 n_processes=56 lib=quimb   time=196.   hyperopt_time=10

n=1000 p=4 d=3 n_edges=1500 n_processes=56 lib=qtensor time=11.11
n=1000 p=4 d=3 n_edges=1500 n_processes=56 lib=quimb   time=>300     hyperopt_time=10

n=1000 p=3 d=3 n_edges=1500 n_processes=56 lib=qtensor time=4.89
n=1000 p=3 d=3 n_edges=1500 n_processes=56 lib=quimb   time=101     hyperopt_time=1
n=1000 p=3 d=3 n_edges=1500 n_processes=56 lib=quimb   time=96     hyperopt_time=0.3
n=1000 p=3 d=3 n_edges=1500 n_processes=56 lib=acqdp   time=N/A

n=1000 p=2 d=3 n_edges=1500 n_processes=56 lib=qtensor time=1.54
n=1000 p=2 d=3 n_edges=1500 n_processes=56 lib=quimb   time=44     hyperopt_time=.3
n=1000 p=2 d=3 n_edges=1500 n_processes=56 lib=quimb   time=31     hyperopt_time=.01
n=1000 p=2 d=3 n_edges=1500 n_processes=56 lib=acqdp   time=260

```
