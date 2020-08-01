#!/bin/bash
pwd
queue=skylake_8180
echo "Starting session to queue $queue..."
job_id=$(qsub -t300 -n1 -q $queue --cwd=$(pwd) qsub_entry.sh)
echo "Qsub Job: $job_id"
logfile=$job_id.output
errfile=$job_id.error

sleep 3
tail --retry -f $logfile &
tail_pid=$!

./job_wait.sh $job_id &
wait $!
echo "Errors:"
cat $errfile
kill $tail_pid
