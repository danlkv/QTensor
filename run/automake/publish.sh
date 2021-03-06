#!/bin/bash

echo "## Automake run result" >> results/result.md

echo "### Performance summary:" >> results/result.md
tail -n 4 time_vs_flops.log >> results/result.md
echo "\n" >> results/result.md

echo "\n" >> results/result.md
echo "Backend used: numpy.einsum" >> results/result.md
echo "\n" >> results/result.md

echo "### Performance plot:" >> results/result.md
cml-publish results/time_vs_flops.png >> results/result.md

echo "\n" >> results/result.md
echo "Run date: $(date)" >> results/result.md
cml-send-comment results/result.md
