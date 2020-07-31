# Atomatic run of qsub jobs


JLSE login has a github runner that will execute any commands.

## Async

If we use qsub asyncronously, we need a script that waits for job completion.

Use `qstat` for that [docs](http://web.mit.edu/longjobs/www/status.html)

- `qstat -u username` will show if we
- `qstat 434` shows. Have to get job id when submitted

`./submit_wait.sh` submits the job using `./entry.sh`, takes the job id and waits for the completion `./wait.sh`


## Sync

`qsub -I -x myscript.sh`
