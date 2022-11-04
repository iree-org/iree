#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Sets up `mmperf` (https://github.com/mmperf/mmperf).
#
# `mmperf` benchmarks matrix-multiply workloads on IREE and other backends such
# as RUY, TVM, Halide, CuBLAS, etc. Some backends are included as submodules
# in the `mmperf` repo and built from source, and other backends are expected
# to already be installed.
#
# Usage:
#    ./setup_mmperf.sh \
#        <mmperf repo dir>

set -xeuo pipefail

export REPO_DIR=$1

pushd ${REPO_DIR}
git clone --recurse-submodules https://github.com/mmperf/mmperf.git
pushd mmperf

# Create virtual environment.
python3 -m venv mmperf.venv
source mmperf.venv/bin/activate
pip install -r requirements.txt
pip install -r ./external/llvm-project/mlir/python/requirements.txt

# Since we are updating the IREE repo at each run, make sure there are no local changes.
pushd external/iree
git restore .
git submodule foreach --recursive git restore .
popd

popd

# Since the root user is used to clone the mmperf repo, we update permissions
# so that a runner can access this repo.
chmod -R 777 .
