#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Sets up `convperf` (https://github.com/nod-ai/convperf).
#
# `convperf` benchmarks convolution workloads on IREE and other backends such
# as libxsmm. IREE is included as a submodule.
#
# Currently x86 CPU is supported.
#
# Usage:
#    ./build_and_run_convperf.sh \
#        <convperf build dir> \
#        <convperf results dir> \
#        <iree commit branch or sha> \
#        <convperf repo dir> (optional)

set -xeuo pipefail

export BUILD_DIR=$1
export RESULTS_DIR=$2
export IREE_COMMIT=${3:-"origin/main"}
export REPO_DIR=${4:-"${CONVPERF_REPO_DIR}"}

pushd ${REPO_DIR}
source convperf.venv/bin/activate

# Set all repos as a safe directory. Since this repo was created in the
# Dockerfile under `root`, git will not run commands on this repo as a
# non-root user unless it is marked safe.
for i in $(find "${REPO_DIR}" -name '.git' | xargs dirname); do
  git config --global --add safe.directory $i
done

# Update IREE.
pushd external/iree
git fetch https://github.com/openxla/iree "${IREE_COMMIT}"
git checkout "${IREE_COMMIT}"
git submodule update --init --jobs 8 --depth 1
popd # external/iree

popd # ${REPO_DIR}

# Build ConvPerf.
cmake -GNinja \
  -DCMAKE_C_COMPILER="${CC:-clang}" \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -B "${BUILD_DIR}" "${REPO_DIR}"
cmake --build "${BUILD_DIR}"

# Run ConvPerf for several threading configurations.
declare -a threads=( 1 2 4 8 16 )

for i in "${threads[@]}"; do
  python3 "${REPO_DIR}/convperf.py" \
      --benchmark_tool "${BUILD_DIR}/tools/benchmark_conv" \
      --runners iree,xsmm \
      --benchmark_sizes \
      "${REPO_DIR}/benchmark_sizes/resnet50.json" \
      --num_threads="$i"

  python "${REPO_DIR}/convperf.py" --visualize --runtimes_file runtimes.json
  mv runtimes.json "${RESULTS_DIR}/resnet50_thread$i.json"
  mv convs.png "${RESULTS_DIR}/resnet50_thread$i.png"
done
