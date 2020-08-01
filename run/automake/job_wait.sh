#!/bin/bash
until ! qstat $1 | grep -m 1 danlkv; do sleep 5; done
