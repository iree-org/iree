#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build tests based on benchmark suites for IREE.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_TARGET_BUILD_DIR, defaulting to
# "build-benchmark-suites-test". Designed for CI, but can be run manually. It
# reuses the build directory if it already exists. Expects to be run from the
# root of the IREE repository.
set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-benchmark-suites-test}}"
E2E_TEST_ARTIFACTS_DIR="$(realpath ${E2E_TEST_ARTIFACTS_DIR:-build-e2e-test-artifacts/e2e_test_artifacts})"
IREE_HOST_BIN_DIR="$(realpath ${IREE_HOST_BIN_DIR})"

source build_tools/cmake/setup_build.sh

# Configure, build, test
declare -a CMAKE_ARGS=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DPYTHON_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
  "-DIREE_BUILD_COMPILER=OFF"
  "-DIREE_HOST_BIN_DIR=${IREE_HOST_BIN_DIR}"
  "-DIREE_BUILD_TESTS=ON"
  "-DIREE_BUILD_SAMPLES=OFF"
  "-DIREE_E2E_TEST_ARTIFACTS_DIR=${E2E_TEST_ARTIFACTS_DIR}"
)

echo "Configuring to build tests for benchmark suites"
"${CMAKE_BIN}" "${CMAKE_ARGS[@]}"

echo "Building tests artifacts"
"${CMAKE_BIN}" --build "${BUILD_DIR}" \
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
ctest --test-dir ${BUILD_DIR} ${ctest_args[@]} \
  --label-regex ${label_include_regex}
