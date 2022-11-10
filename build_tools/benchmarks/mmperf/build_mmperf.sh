#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Builds `mmperf` (https://github.com/mmperf/mmperf).
#
# `mmperf` benchmarks matrix-multiply workloads on IREE and other backends such
# as RUY, TVM, Halide, CuBLAS, etc. Some backends are included as submodules
# in the `mmperf` repo and built from source, and other backends are expected
# to already be installed.
#
# Please refer to `build_tools/docker/mmperf/Dockerfile` for commands on
# installing various backends.
#
# Currently x86 CPU and CUDA are supported.
#
# Usage:
#    ./run_mmperf.sh \
#        <mmperf repo dir> \
#        <mmperf build dir> \
#        <backend> e.g. "cpu", "cuda".

set -xeuo pipefail

export REPO_DIR=$1
export BUILD_DIR=$2
# Either `cpu` or `cuda`.
export BACKEND=$3

pushd ${REPO_DIR}

# Set all repos as a safe directory. Since this repo was created in the
# DockerFile under `root`, git will not run commands on this repo as a
# non-root user unless it is marked safe.
for i in $(find ${REPO_DIR} -name '.git' | xargs dirname); do
  git config --global --add safe.directory $i
done

source mmperf.venv/bin/activate

# Update IREE.
pushd external/iree
git restore .
git submodule foreach --recursive git restore .
git fetch
git checkout origin/main
git submodule update --init --jobs 8 --depth 1
popd

popd

# Build mmperf.
if [ ${BACKEND} == "cuda" ]; then
  cmake \
    -GNinja \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DUSE_IREE=ON \
    -DIREE_CUDA=ON \
    -DUSE_CUBLAS=ON \
    -B ${BUILD_DIR} ${REPO_DIR}
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
    -B ${BUILD_DIR} ${REPO_DIR}
fi

cmake --build ${BUILD_DIR} --verbose
