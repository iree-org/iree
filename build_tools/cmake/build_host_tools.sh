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
CMAKE_BIN="${CMAKE_BIN:-$(which cmake)}"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-OFF}"

source build_tools/cmake/setup_build.sh

mkdir -p "${INSTALL_DIR}"

declare -a CMAKE_ARGS=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"

  "-DCMAKE_INSTALL_PREFIX=$(realpath ${INSTALL_DIR})"
  "-DIREE_ENABLE_LLD=ON"
  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"
  "-DIREE_BUILD_COMPILER=ON"
  "-DIREE_BUILD_TESTS=OFF"
  "-DIREE_BUILD_SAMPLES=OFF"

  # Enable CUDA compiler and runtime builds unconditionally. Our CI images all
  # have enough deps to at least build CUDA support and compile CUDA binaries
  # (but not necessarily test on real hardware).
  "-DIREE_HAL_DRIVER_CUDA=ON"
  "-DIREE_TARGET_BACKEND_CUDA=ON"

  # Enable the WebGPU compiler build. All deps get fetched as needed, but some
  # of the deps are too large to enable by default for all developers.
  "-DIREE_TARGET_BACKEND_WEBGPU=ON"
)

"${CMAKE_BIN}" "${CMAKE_ARGS[@]}"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target install -- -k 0
