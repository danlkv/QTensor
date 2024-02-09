# Running QTensor at HPC scale

Check out the qaoa_amplitude_MPI.py example.
Usage:

```
mpiexec -n 4 python qaoa_amplitude_MPI.py
```

It helps to understand the contraction width of your particular problem, which can inform what `max_tw` parameter you should use. 
The relation between number of tasks run in paralell (n_tasks), max_tw and contraction width is following: `log_2(n_tasks) >= width - max_tw`.
