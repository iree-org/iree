#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build only the IREE runtime using CMake for the host
#
# Designed for CI, but can be run locally. The desired build directory can be
# passed as the first argument. Otherwise, it uses the environment variable
# IREE_TARGET_BUILD_DIR, defaulting to "build-runtime". It reuses the build
# directory if it already exists. Expects to be run from the root of the IREE
# repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-runtime}}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
IREE_BUILD_PYTHON_BINDINGS="${IREE_BUILD_PYTHON_BINDINGS:-ON}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

declare -a args
args=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"

  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DIREE_BUILD_COMPILER=OFF"
  "-DIREE_ENABLE_ASSERTIONS=ON"
  "-DIREE_BUILD_SAMPLES=ON"

  # Use `lld` for faster linking.
  "-DIREE_ENABLE_LLD=ON"

  "-DIREE_BUILD_PYTHON_BINDINGS=${IREE_BUILD_PYTHON_BINDINGS}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DPYTHON_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
)
"${CMAKE_BIN}" "${args[@]}"

"${CMAKE_BIN}" --build "${BUILD_DIR}" -- -k 0

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi
