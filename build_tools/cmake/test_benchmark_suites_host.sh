#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build tests based on benchmark suites for IREE.
# Designed for CI, but can be run locally.
set -xeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}"

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
BUILD_BENCHMARK_SUITE_DIR="$(realpath ${BUILD_BENCHMARK_SUITE_DIR:-$ROOT_DIR/build-benchmarks/benchmark_suites})"
IREE_HOST_BINARY_ROOT="$(realpath ${IREE_HOST_BINARY_ROOT})"
BUILD_HOST_DIR="${BUILD_HOST_DIR:-build-benchmark-suites-test}"

"$CMAKE_BIN" --version
ninja --version

# --------------------------------------------------------------------------- #
if [[ -d "${BUILD_HOST_DIR}" ]]; then
  echo "${BUILD_HOST_DIR} directory already exists. Will use cached results there."
else
  echo "${BUILD_HOST_DIR} directory does not already exist. Creating a new one."
  mkdir -p "${BUILD_HOST_DIR}"
fi

# Configure, build, test
declare -a CMAKE_ARGS=(
  "-G" "Ninja"
  "-B" "${BUILD_HOST_DIR}"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
  "-DIREE_BUILD_COMPILER=OFF"
  "-DIREE_HOST_BINARY_ROOT=${IREE_HOST_BINARY_ROOT}"
  "-DIREE_BUILD_TESTS=ON"
  "-DIREE_BUILD_SAMPLES=OFF"
  "-DIREE_BENCHMARK_SUITE_DIR=${BUILD_BENCHMARK_SUITE_DIR}"
)

echo "Configuring to build tests for benchmark suites"
"${CMAKE_BIN}" "${CMAKE_ARGS[@]}"

echo "Building tests artifacts"
"${CMAKE_BIN}" --build "${BUILD_HOST_DIR}" \
  --target iree-run-module iree-run-module-test-deps -- -k 0

ctest_args=(
  "--timeout 900"
  "--output-on-failure"
  "--no-tests=error"
)

declare -a label_args=(
  "^test-type=run-module-test$"
)
label_include_regex="($(IFS="|" ; echo "${label_args[*]}"))"

echo "******** Running run-module CTest ********"
ctest --test-dir ${BUILD_HOST_DIR} ${ctest_args[@]} \
  --label-regex ${label_include_regex}
