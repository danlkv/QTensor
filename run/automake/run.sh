pwd
echo "Starting session to qsub..."
qsub -I -cwd -x qsub_entry.sh
