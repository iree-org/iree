#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the project towards RISCV with CMake using Kokoro.

set -xeuo pipefail

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Check these exist and print the versions for later debugging
export CMAKE_BIN="$(which cmake)"
"${CMAKE_BIN}" --version
"${CC}" --version
"${CXX}" --version
python3 --version

echo "Initializing submodules"
git submodule update --init --jobs 8 --depth 1

export ROOT_DIR="$(git rev-parse --show-toplevel)"
export BUILD_HOST_DIR="${ROOT_DIR}/build-host"
export BUILD_RISCV_DIR="${ROOT_DIR}/build-riscv"
export LLVM_BINDIR="${BUILD_HOST_DIR}/third_party/llvm-project/llvm/bin"

# BUILD the integrations iree-import-tflite with Bazel
echo "Build iree-import-tflite with Bazel"
pushd "${ROOT_DIR}/integrations/tensorflow"
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
# Yes this is really the best way to get the path to a single binary
export IREE_IMPORT_TFLITE_BIN="$(
"${BAZEL_CMD[@]}" run --run_under=echo \
  --config=remote_cache_bazel_ci \
  --config=generic_clang \
  //iree_tf_compiler:iree-import-tflite
)"
popd


echo "Cross-compiling with cmake"
./build_tools/cmake/build_host_and_riscv.sh

echo "Run sanity tests"
./build_tools/testing/test_riscv64.sh
