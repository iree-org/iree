#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build e2e test artifacts using a host tools directory.
# Designed for CI, but can be run locally.
#
# This is copied and modified from build_tools/cmake/build_benchmarks.sh.
# We will remove build_tools/cmake/build_benchmarks.sh once everything has been
# migrated.

set -xeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}"

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
IREE_HOST_BINARY_ROOT="$(realpath ${IREE_HOST_BINARY_ROOT})"
IREE_TF_BINARIES_DIR="${IREE_TF_BINARIES_DIR:-integrations/tensorflow/bazel-bin/iree_tf_compiler}"
BUILD_E2E_TEST_ARTIFACTS_DIR="${BUILD_E2E_TEST_ARTIFACTS_DIR:-$ROOT_DIR/build-e2e-test-artifacts}"

"$CMAKE_BIN" --version
ninja --version

if [[ -d "${BUILD_E2E_TEST_ARTIFACTS_DIR}" ]]; then
  echo "${BUILD_E2E_TEST_ARTIFACTS_DIR} directory already exists. Will use cached results there."
else
  echo "${BUILD_E2E_TEST_ARTIFACTS_DIR} directory does not already exist. Creating a new one."
  mkdir "${BUILD_E2E_TEST_ARTIFACTS_DIR}"
fi

echo "Configuring to build e2e test artifacts"
"${CMAKE_BIN}" -B "${BUILD_E2E_TEST_ARTIFACTS_DIR}" \
  -G Ninja \
  -DIREE_HOST_BINARY_ROOT="${IREE_HOST_BINARY_ROOT}" \
  -DIREE_BUILD_EXPERIMENTAL_E2E_TEST_ARTIFACTS=ON \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_IMPORT_TFLITE_PATH="${IREE_TF_BINARIES_DIR}/iree-import-tflite" \
  -DIREE_IMPORT_TF_PATH="${IREE_TF_BINARIES_DIR}/iree-import-tf"

echo "Building e2e test artifacts"
"${CMAKE_BIN}" \
  --build "${BUILD_E2E_TEST_ARTIFACTS_DIR}" \
  --target iree-e2e-test-artifacts \
  -- -k 0
