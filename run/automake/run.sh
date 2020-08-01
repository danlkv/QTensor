#!/bin/bash
pwd
queue=skylake_8180
echo "Starting session to queue `$queue`..."
job_id=$(qsub -t300 -n1 -q $queue qsub_entry.sh)
ecskylake_8180ho "Qsub Job: $job_id"
logfile=$job_id.output

sleep 1
tail -f $logfile &
tail_pid=$!

./job_wait.sh $job_id &
wait $!
kill $tail_pid
