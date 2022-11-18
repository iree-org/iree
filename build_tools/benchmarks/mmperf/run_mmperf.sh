#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs `mmperf` (https://github.com/mmperf/mmperf).
#
# `mmperf` benchmarks matrix-multiply workloads on IREE and other backends such
# as RUY, TVM, Halide, CuBLAS, etc. Some backends are included as submodules
# in the `mmperf` repo and built from source, and other backends are expected
# to already be installed.
#
# Please refer to `build_tools/docker/mmperf/Dockerfile` for commands on
# installing various backends.
#
# Usage:
#    ./run_mmperf.sh \
#        <mmperf repo dir> \
#        <mmperf build dir> \
#        <results directory>

set -xeuo pipefail

export REPO_DIR=$1
export BUILD_DIR=$2
export REPORT_DIR=$3

source ${REPO_DIR}/mmperf.venv/bin/activate

# Run benchmark.
python3 ${REPO_DIR}/mmperf.py ${BUILD_DIR}/matmul/ ${REPORT_DIR}
