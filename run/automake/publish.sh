#!/bin/bash

echo "## Automake run result" >> results/result.md
cml-publish results/image.png >> results/result.md
echo "\n" >> results/result.md
echo "Run date: $(date)" >> results/result.md
cml-send-comment results/result.md
