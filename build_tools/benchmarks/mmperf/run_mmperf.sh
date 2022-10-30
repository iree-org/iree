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
#    ./run_shark.sh \
#        <SHA of https://github.com/nod-ai/SHARK.git to pin to> \
#        <pytest regex> e.g. "cpu", "cuda", "cuda and torch".
#        <driver> e.g. "cpu", "cuda", "vulkan"
#        <output directory>

set -xeuo pipefail

export REPORT_DIR=$1
# Either `cpu` or `cuda`.
export BACKEND=$2

git clone https://github.com/mmperf/mmperf.git
cd mmperf

# Update IREE.
cd external/iree
git fetch --all
git checkout origin/main
git submodule update --init --recursive
cd -

# Create virtual environment.
python3 -m venv mmperf.venv
source mmperf.venv/bin/activate
pip install -r requirements.txt
pip install -r ./external/llvm-project/mlir/python/requirements.txt
pip install requests

# Build mmperf.
if [ ${BACKEND} == "cuda" ]; then
cmake \
  -GNinja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DUSE_IREE=ON \
  -DIREE_CUDA=ON \
  -DUSE_CUBLAS=ON \
  -B ../mmperf-build .
else
MKL_DIR=/opt/intel/mkl BLIS_DIR=/opt/blis cmake \
  -GNinja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DMKL_DIR=/opt/intel/mkl \
  -DBLIS_DIR=/opt/blis \
  -DUSE_MKL=ON \
  -DUSE_RUY=ON \
  -DUSE_IREE=ON \
  -DIREE_LLVMCPU=ON \
  -DUSE_HALIDE=ON \
  -DUSE_OPENBLAS=ON \
  -DUSE_BLIS=ON \
  -DUSE_TVM=ON \
  -B ../mmperf-build .
fi
cmake --build ../mmperf-build -j32 --verbose

# Run benchmark.
cd ../
python3 mmperf/mmperf.py ./mmperf-build/matmul/ ${REPORT_DIR}
