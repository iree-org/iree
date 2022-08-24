#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build benchmark suites for IREE using an *already built* build directory.
# Designed for CI, but can be run locally.
# TODO(#4662): this should just allow passing in host tools.

set -xeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}"

BUILD_DIR="${1:-${IREE_BUILD_DIR:-build}}"
IREE_HOST_BINARY_ROOT="${IREE_HOST_BINARY_ROOT:-build-host/install}"
CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
IREE_TF_BINARIES_DIR="${IREE_TF_BINARIES_DIR:-integrations/tensorflow/bazel-bin/iree_tf_compiler}"

"$CMAKE_BIN" --version
ninja --version

echo "Reconfiguring to enable benchmarks"
# Note that we're relying on CMake caching here
"${CMAKE_BIN}" -B "${BUILD_DIR}" \
  -G Ninja \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_HOST_BINARY_ROOT="${IREE_HOST_BINARY_ROOT}" \
  -DIREE_BUILD_BENCHMARKS=ON \
  -DIREE_BUILD_MICROBENCHMARKS=ON \
  -DIREE_IMPORT_TFLITE_PATH="${IREE_TF_BINARIES_DIR}/iree-import-tflite" \
  -DIREE_IMPORT_TF_PATH="${IREE_TF_BINARIES_DIR}/iree-import-tf"

echo "Building benchmark artifacts"
"${CMAKE_BIN}" \
  --build "${BUILD_DIR}" \
  --target iree-benchmark-suites iree-microbenchmark-suites \
  -- -k 0
