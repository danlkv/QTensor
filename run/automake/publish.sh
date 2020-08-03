#!/bin/bash

echo "## Automake run result" >> results/result.md
cml-publish results/figure.png >> results/result.md
echo "\n" >> results/result.md

echo "## Profile" >> results/result.md
cat results/profile.txt >> results/result.md
echo "\n" >> results/result.md
echo "Run date: $(date)" >> results/result.md
cml-send-comment results/result.md
