#!/bin/bash

SP=$1
MBPS=$2

echo ">>> [SERVER] Running with SP=$SP, MBPS=$MBPS"
python experiment/server_run.py -SP=$SP --mbps=$MBPS