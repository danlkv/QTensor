#!/bin/bash
pwd
echo "Starting session to qsub..."
job_id=$(qsub -I -x qsub_entry.sh)
echo "Qsub Job: $job_id"
logfile=$job_id.output

tail -f $logfile &
tail_pid=$!

./job_wait.sh $job_id &
wait $!
kill $tail_pid
