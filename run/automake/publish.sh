#!/bin/bash

echo "## Automake run result" >> results/result.md

echo "### Performance summary:" >> results/result.md
tail -n 5 results/time_vs_flops.txt >> results/result.md
echo "\n" >> results/result.md

echo "\n" >> results/result.md
echo "Backend used: mkl (full)" >> results/result.md
echo "\n" >> results/result.md

echo "### Performance plot:" >> results/result.md
cml-publish results/time_vs_flops.png >> results/result.md

echo "\n" >> results/result.md
echo "Run date: $(date)" >> results/result.md
cml-send-comment results/result.md
