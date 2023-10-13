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
BUILD_PRESET="${BUILD_PRESET:-test}"

source build_tools/cmake/setup_build.sh
# Note: not using ccache since the runtime build should be fast already.

declare -a args
args=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DPYTHON_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
  "-DIREE_BUILD_COMPILER=OFF"
)

case "${BUILD_PRESET}" in
  test)
    args+=(
      -DIREE_ENABLE_ASSERTIONS=ON
      -DIREE_BUILD_SAMPLES=ON
    )
    ;;
  benchmark)
    args+=(
      -DIREE_ENABLE_ASSERTIONS=OFF
      -DIREE_BUILD_SAMPLES=OFF
      -DIREE_BUILD_TESTS=OFF
    )
    ;;
  benchmark-with-tracing)
    args+=(
      -DIREE_ENABLE_ASSERTIONS=OFF
      -DIREE_BUILD_SAMPLES=OFF
      -DIREE_BUILD_TESTS=OFF
      -DIREE_ENABLE_RUNTIME_TRACING=ON
    )
    ;;
  *)
    echo "Unknown build preset: ${BUILD_PRESET}"
    exit 1
    ;;
esac

"${CMAKE_BIN}" "${args[@]}"

case "${BUILD_PRESET}" in
  test)
    "${CMAKE_BIN}" --build "${BUILD_DIR}" -- -k 0
    ;;
  benchmark|benchmark-with-tracing)
    "${CMAKE_BIN}" --build "${BUILD_DIR}" --target iree-benchmark-module -- -k 0
    ;;
  *)
    echo "Unknown build preset: ${BUILD_PRESET}"
    exit 1
    ;;
esac
