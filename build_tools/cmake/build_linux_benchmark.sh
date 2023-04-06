#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

# Print the UTC time when set -x is on.
export PS4='[$(date -u "+%T %Z")] '

# Check these exist and print the versions for later debugging.
CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
"${CMAKE_BIN}" --version
"${CC}" --version
"${CXX}" --version
ninja --version
python3 --version

echo "Initializing submodules"
git submodule sync
git submodule update --init --jobs 8 --depth 1

ROOT_DIR=$(git rev-parse --show-toplevel)
cd "${ROOT_DIR}"

# BUILD the iree-import-tflite binary for importing models to benchmark from
# TFLite FlatBuffers.
cd "${ROOT_DIR}/integrations/tensorflow"
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
BAZEL_BINDIR="$(${BAZEL_CMD[@]} info bazel-bin)"
"${BAZEL_CMD[@]}" build \
      //iree_tf_compiler:iree-import-tflite \
      //iree_tf_compiler:iree-import-tf \
      --config=generic_clang \
      --config=remote_cache_bazel_tf_ci
# So the benchmark build below can find the importer binaries that were built.
export PATH="$PWD/bazel-bin/iree_tf_compiler:$PATH"

# --------------------------------------------------------------------------- #
# Build for the host.

cd "${ROOT_DIR}"

if [ -d "build-host" ]
then
  echo "build-host directory already exists. Will use cached results there."
else
  echo "build-host directory does not already exist. Creating a new one."
  mkdir build-host
fi
cd build-host

# Configure, build, install.
"${CMAKE_BIN}" -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_LEGACY_BENCHMARKS=ON \
  -DIREE_ENABLE_LEGACY_COMPILATION_BENCHMARKS=ON \
  -DIREE_BUILD_MICROBENCHMARKS=ON \
  -DIREE_BUILD_SAMPLES=OFF

"${CMAKE_BIN}" --build . --target install -- -k 0
"${CMAKE_BIN}" --build . --target iree-benchmark-suites-linux-x86_64 -- -k 0
"${CMAKE_BIN}" --build . --target iree-microbenchmark-suites -- -k 0
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Build for the target (linux-x86_64).

cd "${ROOT_DIR}"

if [ -d "build-linux-x86_64" ]
then
  echo "build-linux-x86_64 directory already exists. Will use cached results there."
else
  echo "build-linux-x86_64 directory does not already exist. Creating a new one."
  mkdir build-linux-x86_64
fi
cd build-linux-x86_64

"${CMAKE_BIN}" -G Ninja .. \
  -DIREE_HOST_BIN_DIR="${PWD}/../build-host/install/bin" \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_BUILD_SAMPLES=OFF
"${CMAKE_BIN}" --build . --target iree-benchmark-module -- -k 0

# --------------------------------------------------------------------------- #
