#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build e2e test artifacts using a host tools directory.
#
# This is copied and modified from build_tools/cmake/build_benchmarks.sh. We
# will remove build_tools/cmake/build_benchmarks.sh once everything has been
# migrated.
#
# The required IREE_HOST_BIN_DIR environment variable indicates the location
# of the precompiled IREE binaries.
#
# Designed for CI, but can be run locally. The desired build directory can be
# passed as the first argument. Otherwise, it uses the environment variable
# IREE_BUILD_E2E_TEST_ARTIFACTS_DIR, defaulting to "build-e2e-test-artifacts".
# It reuses the build directory if it already exists. Expects to be run from the
# root of the IREE repository.


set -xeuo pipefail

BUILD_DIR="${1:-${IREE_BUILD_E2E_TEST_ARTIFACTS_DIR:-build-e2e-test-artifacts}}"
IREE_HOST_BIN_DIR="$(realpath ${IREE_HOST_BIN_DIR})"
IREE_TF_BINARIES_DIR="${IREE_TF_BINARIES_DIR:-integrations/tensorflow/bazel-bin/iree_tf_compiler}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_local_python.sh

echo "Configuring to build e2e test artifacts"
"${CMAKE_BIN}" -B "${BUILD_DIR}" \
  -G Ninja \
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DIREE_HOST_BIN_DIR="${IREE_HOST_BIN_DIR}" \
  -DIREE_BUILD_E2E_TEST_ARTIFACTS=ON \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_IMPORT_TFLITE_PATH="${IREE_TF_BINARIES_DIR}/iree-import-tflite"

echo "Building e2e test artifacts"
"${CMAKE_BIN}" \
  --build "${BUILD_DIR}" \
  --target iree-e2e-test-artifacts \
  -- -k 0
