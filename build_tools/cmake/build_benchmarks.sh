#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build benchmark suites for IREE using a host tools directory.
# Designed for CI, but can be run locally.

set -xeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}"

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
IREE_HOST_BINARY_ROOT="$(realpath ${IREE_HOST_BINARY_ROOT})"
IREE_TF_BINARIES_DIR="${IREE_TF_BINARIES_DIR:-integrations/tensorflow/bazel-bin/iree_tf_compiler}"
BUILD_BENCHMARKS_DIR="${BUILD_BENCHMARKS_DIR:-$ROOT_DIR/build-benchmarks}"

"$CMAKE_BIN" --version
ninja --version

if [[ -d "${BUILD_BENCHMARKS_DIR}" ]]; then
  echo "${BUILD_BENCHMARKS_DIR} directory already exists. Will use cached results there."
else
  echo "${BUILD_BENCHMARKS_DIR} directory does not already exist. Creating a new one."
  mkdir "${BUILD_BENCHMARKS_DIR}"
fi

echo "Configuring to build benchmarks"
"${CMAKE_BIN}" -B "${BUILD_BENCHMARKS_DIR}" \
  -G Ninja \
  -DIREE_HOST_BINARY_ROOT="${IREE_HOST_BINARY_ROOT}" \
  -DIREE_BUILD_BENCHMARKS=ON \
  -DIREE_BUILD_MICROBENCHMARKS=ON \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_IMPORT_TFLITE_PATH="${IREE_TF_BINARIES_DIR}/iree-import-tflite" \
  -DIREE_IMPORT_TF_PATH="${IREE_TF_BINARIES_DIR}/iree-import-tf"

echo "Building benchmark artifacts"
"${CMAKE_BIN}" \
  --build "${BUILD_BENCHMARKS_DIR}" \
  --target iree-benchmark-suites iree-microbenchmark-suites \
  -- -k 0
