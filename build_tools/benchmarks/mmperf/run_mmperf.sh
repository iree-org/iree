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
#        <mmperf build dir> \
#        <results directory> \
#        <mmperf repo dir> (optional)

set -xeuo pipefail

export BUILD_DIR=$1
export REPORT_DIR=$2
export REPO_DIR=${3:-${MMPERF_REPO_DIR}}

source ${REPO_DIR}/mmperf.venv/bin/activate

# Run benchmark.
python3 ${REPO_DIR}/mmperf.py ${BUILD_DIR}/matmul/ ${REPORT_DIR}
