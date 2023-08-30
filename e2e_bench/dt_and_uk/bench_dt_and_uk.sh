#!/bin/bash

# The script will find iree tools in PATH. To reproduce data-tiling benchmarks,
# please build tools at 4cc440bc3599207828585f4b51b685a1585fe431

export MODEL_DIR=..

export IREE_BENCHMARK_MODULE="iree-benchmark-module"
export TRACE_MODE=0

THREADS=1 ../run.sh | tee run1.log
THREADS=4 ../run.sh | tee run4.log
THREADS=8 ../run.sh | tee run8.log

# export IREE_BENCHMARK_MODULE="iree-traced-benchmark-module"
# export TRACE_MODE=1
# 
# THREADS=1 ../run.sh | tee traced_run1.log
