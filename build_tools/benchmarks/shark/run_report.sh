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
SUMMARY_PATH=$2

python3 -m venv iree.venv
source iree.venv/bin/activate
pip install --upgrade pip
pip install pandas
pip install jinja2

python3 ./build_tools/benchmarks/reporting/parse_shark_benchmarks.py \
  --shark_version="tbd" \
  --iree_version="tbd" \
  --cpu_shark_csv="${RAW_RESULTS_DIR}/cpu_shark.csv" \
  --cpu_iree_csv="${RAW_RESULTS_DIR}/cpu_iree.csv" \
  --cpu_baseline_csv="${RAW_RESULTS_DIR}/cpu_iree.csv" \
  --gpu_shark_csv="${RAW_RESULTS_DIR}/cuda_shark.csv" \
  --gpu_iree_csv="${RAW_RESULTS_DIR}/cuda_iree.csv" \
  --gpu_baseline_csv="${RAW_RESULTS_DIR}/cuda_iree.csv" \
  --output_path="${SUMMARY_PATH}"
