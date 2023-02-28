#!/bin/bash

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build "all" of the IREE project.
#
# Designed for CI, but can be run locally. The desired build directory can be
# passed as the first argument. Otherwise, it uses the environment variable
# IREE_BUILD_DIR, defaulting to "build". It reuses the build directory if it
# already exists. Expects to be run from the root of the IREE repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_BUILD_DIR:-build}}"
INSTALL_DIR="${IREE_INSTALL_DIR:-${BUILD_DIR}/install}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"
# Enable WebGPU compiler builds and tests by default. All deps get fetched as
# needed, but some of the deps are too large to enable by default for all
# developers.
IREE_TARGET_BACKEND_WEBGPU="${IREE_TARGET_BACKEND_WEBGPU:-ON}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

declare -a CMAKE_ARGS=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"

  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DCMAKE_INSTALL_PREFIX=$(realpath ${INSTALL_DIR})"
  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"

  # Use `lld` for faster linking.
  "-DIREE_ENABLE_LLD=ON"

  # Enable docs build on the CI. The additional builds are pretty fast and
  # give us early warnings for some types of website publication errors.
  "-DIREE_BUILD_DOCS=ON"

  # Enable building the python bindings on CI.
  "-DIREE_BUILD_PYTHON_BINDINGS=ON"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DPYTHON_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"

  "-DIREE_TARGET_BACKEND_WEBGPU=${IREE_TARGET_BACKEND_WEBGPU}"
)

"$CMAKE_BIN" "${CMAKE_ARGS[@]}"
echo "Building all"
echo "------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" -- -k 0

echo "Building 'install'"
echo "------------------"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target install -- -k 0

echo "Building test deps"
echo "------------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" --target iree-test-deps -- -k 0

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi
