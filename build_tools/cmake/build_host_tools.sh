#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds IREE host tools for use by other builds.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_HOST_BUILD_DIR, defaulting to "build-host".
# Designed for CI, but can be run manually. It reuses the build directory if it
# already exists. Expects to be run from the root of the IREE repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_HOST_BUILD_DIR:-build-host}}"
INSTALL_DIR="${INSTALL_DIR:-${BUILD_DIR}/install}"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-OFF}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

mkdir -p "${INSTALL_DIR}"

declare -a CMAKE_ARGS=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DPYTHON_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"

  "-DCMAKE_INSTALL_PREFIX=$(realpath ${INSTALL_DIR})"
  "-DIREE_ENABLE_LLD=ON"
  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"
  "-DIREE_BUILD_COMPILER=ON"
  "-DIREE_BUILD_TESTS=OFF"
  "-DIREE_BUILD_SAMPLES=OFF"

  # Enable the WebGPU compiler build. All deps get fetched as needed, but some
  # of the deps are too large to enable by default for all developers.
  "-DIREE_TARGET_BACKEND_WEBGPU_SPIRV=ON"
)

"${CMAKE_BIN}" "${CMAKE_ARGS[@]}"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target install -- -k 0

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi
