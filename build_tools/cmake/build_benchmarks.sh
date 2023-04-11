#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build benchmark suites using a host tools directory.
#
# The required IREE_HOST_BIN_DIR environment variable indicates the location
# of the precompiled IREE binaries.
#
# Designed for CI, but can be run locally. The desired build directory can be
# passed as the first argument. Otherwise, it uses the environment variable
# IREE_BUILD_BENCHMARKS_DIR, defaulting to "build-benchmarks". It reuses the
# build directory if it already exists. Expects to be run from the root of the
# IREE repository.


set -xeuo pipefail

BUILD_DIR="${1:-${IREE_BUILD_BENCHMARKS_DIR:-build-benchmarks}}"
IREE_HOST_BIN_DIR="$(realpath ${IREE_HOST_BIN_DIR})"
IREE_TF_BINARIES_DIR="${IREE_TF_BINARIES_DIR:-integrations/tensorflow/bazel-bin/iree_tf_compiler}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_local_python.sh

echo "Configuring to build benchmarks"
"${CMAKE_BIN}" -B "${BUILD_DIR}" \
  -G Ninja \
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DIREE_HOST_BIN_DIR="${IREE_HOST_BIN_DIR}" \
  -DIREE_BUILD_LEGACY_BENCHMARKS=ON \
  -DIREE_BUILD_MICROBENCHMARKS=ON \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_IMPORT_TFLITE_PATH="${IREE_TF_BINARIES_DIR}/iree-import-tflite"

echo "Building benchmark artifacts"
"${CMAKE_BIN}" \
  --build "${BUILD_DIR}" \
  --target iree-benchmark-suites iree-microbenchmark-suites \
  -- -k 0
