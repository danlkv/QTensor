#!/bin/bash
while qstat $1 | grep -m 1 $1; do sleep 5; done
