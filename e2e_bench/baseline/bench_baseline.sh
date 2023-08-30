#!/bin/bash

# The script will find iree tools in PATH. To reproduce baseline benchmarks,
# please build tools at 40794933d45fdbb05d631c9612dc91cc343d1efe.

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
