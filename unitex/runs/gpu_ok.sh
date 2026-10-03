#!/bin/bash
# usage: gpu_ok.sh <gpu>   exit 0 when the GPU is free for my jobs.
# GPU 0 also holds Kiran's web app (pid 3608843, a 53 MiB graphics context). It is never touched. GPU 0
# counts as free while that is the only process on it and usage stays under 200 MiB.
g=$1
KIRAN_PID=3608843
used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
pids=$(nvidia-smi -i $g | awk '/Processes:/ {p = 1} p && $2 ~ /^[0-9]+$/ && $5 ~ /^[0-9]+$/ {print $5}')
others=$(echo "$pids" | grep -v -x "$KIRAN_PID" | grep -c .)
if [ "$g" = 0 ]; then
  [ "$used" -lt 200 ] && [ "$others" -eq 0 ]
else
  [ "$used" -lt 30 ] && [ -z "$pids" ]
fi
