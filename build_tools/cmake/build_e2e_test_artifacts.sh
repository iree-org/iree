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
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-e2e-test-artifacts}"
IREE_HOST_BINARY_ROOT="$(realpath ${IREE_HOST_BINARY_ROOT})"
IREE_TF_BINARIES_DIR="${IREE_TF_BINARIES_DIR:-integrations/tensorflow/bazel-bin/iree_tf_compiler}"

cd "${ROOT_DIR}"
source build_tools/cmake/setup_build.sh

echo "Configuring to build e2e test artifacts"
"${CMAKE_BIN}" -B "${BUILD_DIR}" \
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
  --build "${BUILD_DIR}" \
  --target iree-e2e-test-artifacts \
  -- -k 0
