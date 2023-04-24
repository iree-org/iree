#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Benchmarks XLA workloads.
# Expects environment variables `TENSORFLOW_VERSION` and `RUN_HLO_MODULE_PATH` to be set.
#
# Usage:
#    ./benchmark_xla.sh <tensorflow version> <results dir>

set -xeuo pipefail

export TENSORFLOW_VERSION=$1
export RESULTS_DIR=$2

# Build Tensorflow from source.
sudo wget https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64 -O /usr/local/bin/bazel 
sudo chmod +x /usr/local/bin/bazel

export TF_BRANCH=$(echo ${TENSORFLOW_VERSION} | sed "s/\([[:digit:]]\+\.[[:digit:]]\+\)\..*/\1/g")
git clone https://github.com/tensorflow/tensorflow.git
pushd tensorflow
git checkout "r${TF_BRANCH}"
bazel build -c opt --config=cuda tensorflow/compiler/xla/tools/run_hlo_module
export RUN_HLO_MODULE_PATH="$(pwd)/bazel-bin/tensorflow/compiler/xla/tools/run_hlo_module"
popd

# Benchmark workloads.
git clone https://github.com/iree-org/iree-samples.git
pushd iree-samples/iree-tf/benchmark
./benchmark_all.sh "cuda" "${TENSORFLOW_VERSION}" "${RESULTS_DIR}/tf_cuda.csv" "${RUN_HLO_MODULE_PATH}"
cat "${RESULTS_DIR}/tf_cuda.csv"
popd
