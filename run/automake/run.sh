#!/bin/bash
pwd
queue=skylake_8180
echo "Starting session to queue $queue..."
job_id=$(qsub -v -t300 -n1 -q $queue --cwd=$(pwd) qsub_entry.sh)
echo "Qsub Job: $job_id"
logfile=$job_id.output
errfile=$job_id.error

sleep 3
tail --retry -f $logfile &
tail_pid=$!

./job_wait.sh $job_id &
wait $!
kill $tail_pid

if cat $job_id.cobaltlog | grep norbally with an exit code of 0; then
    echo "Job returned with exit code 0"
    exit 0
else
    echo "########################"
    echo "Errors:"
    cat $errfile
    echo "########################"
    echo "Debug log:"
    cat $job_id.cobaltlog
fi
