#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs SHARK tank using both SHARK-Runtime and IREE-Runtime, producing benchmark
# numbers.
#
# Usage:
#    ./run_report.sh \
#        <RAW_RESULTS_DIR> The directory consisting of the csv files to process. \
#        <SUMMARY_PATH> The path of the output HTML file.

set -xeuo pipefail

RAW_RESULTS_DIR=$1
CPU_BASELINE=$2
GPU_BASELINE=$3
SUMMARY_PATH=$4

python3 -m venv iree.venv
source iree.venv/bin/activate
pip install --upgrade pip
pip install pandas
pip install jinja2

export SHARK_VERSION=`ls ${RAW_RESULTS_DIR} | grep "cpu_shark" | sed -e "s/^cpu_shark_\(.*\).csv$/\1/g"`
export IREE_VERSION=`ls ${RAW_RESULTS_DIR} | grep "cpu_iree" | sed -e "s/^cpu_iree_\(.*\).csv$/\1/g"`

python3 ./build_tools/benchmarks/reporting/parse_shark_benchmarks.py \
  --shark_version="${SHARK_VERSION}" \
  --iree_version="${IREE_VERSION}" \
  --cpu_shark_csv="${RAW_RESULTS_DIR}/cpu_shark_${SHARK_VERSION}.csv" \
  --cpu_iree_csv="${RAW_RESULTS_DIR}/cpu_iree_${IREE_VERSION}.csv" \
  --cpu_baseline_csv="${CPU_BASELINE}" \
  --gpu_shark_csv="${RAW_RESULTS_DIR}/cuda_shark_${SHARK_VERSION}.csv" \
  --gpu_iree_csv="${RAW_RESULTS_DIR}/cuda_iree_${IREE_VERSION}.csv" \
  --gpu_baseline_csv="${GPU_BASELINE}" \
  --output_path="${SUMMARY_PATH}"
