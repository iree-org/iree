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
# Please refer to `build_tools/docker/dockerfiles/mmperf.Dockerfile` for commands on
# installing various backends.
#
# Currently x86 CPU and CUDA are supported.
#
# Usage:
#    ./build_mmperf.sh \
#        <mmperf build dir> \
#        <backend> e.g. "cpu", "cuda". \
#        <iree sha> (optional) \
#        <mmperf repo dir> (optional)

set -xeuo pipefail

export BUILD_DIR=$1
# Either `cpu` or `cuda`.
export BACKEND=$2
export IREE_SHA=${3:-"origin/main"}
export REPO_DIR=${4:-${MMPERF_REPO_DIR}}

pushd ${REPO_DIR}
source mmperf.venv/bin/activate

# Set all repos as a safe directory. Since this repo was created in the
# DockerFile under `root`, git will not run commands on this repo as a
# non-root user unless it is marked safe.
for i in $(find ${REPO_DIR} -name '.git' | xargs dirname); do
  git config --global --add safe.directory $i
done

# Update IREE.
pushd external/iree
git fetch https://github.com/openxla/iree "${IREE_SHA}"
git checkout "${IREE_SHA}"
git submodule update --init --jobs 8 --depth 1
popd # external/iree

popd # ${REPO_DIR}

# Build mmperf.
declare -a args=(
  -GNinja
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++
  -DCMAKE_C_COMPILER=/usr/bin/clang
  -DUSE_IREE=ON
)

if [[ ${BACKEND} == "cuda" ]]; then
  args+=(
    -DIREE_CUDA=ON
    -DUSE_CUBLAS=ON
  )
  cmake "${args[@]}" -B ${BUILD_DIR} ${REPO_DIR}
elif [[ ${BACKEND} == "cpu" ]]; then
  args+=(
    -DMKL_DIR=/opt/intel/mkl
    -DBLIS_DIR=/opt/blis
    -DUSE_MKL=ON
    -DUSE_RUY=ON
    -DIREE_LLVMCPU=ON
    -DUSE_HALIDE=ON
    -DUSE_OPENBLAS=ON
    -DUSE_BLIS=ON
    -DUSE_TVM=ON
  )
  MKL_DIR=${MKL_DIR} BLIS_DIR=${BLIS_DIR} cmake "${args[@]}" -B ${BUILD_DIR} ${REPO_DIR}
else
  echo "Error: Unsupported backend."
  exit 1
fi

cmake --build ${BUILD_DIR} --verbose
