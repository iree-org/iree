#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the IREE project towards Android with CMake. Designed for CI,
# but can be run manually.
#
# Requires pre-compiled IREE and TF integrations host tools. Also requires that
# ANDROID_ABI and ANDROID_NDK variables be set

set -xeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}"

CMAKE_BIN="${CMAKE_BIN:-$(which cmake)}"
IREE_HOST_BINARY_ROOT="$(realpath ${IREE_HOST_BINARY_ROOT})"
BUILD_ANDROID_DIR="${BUILD_ANDROID_DIR:-$ROOT_DIR/build-android}"
BUILD_BENCHMARK_SUITE_DIR="${BUILD_BENCHMARK_SUITE_DIR:-$ROOT_DIR/build-benchmarks/benchmark_suites}"
BUILD_PRESET="${BUILD_PRESET:-test}"


if [[ -d "${BUILD_ANDROID_DIR}" ]]; then
  echo "${BUILD_ANDROID_DIR} directory already exists. Will use cached results there."
else
  echo "${BUILD_ANDROID_DIR} directory does not already exist. Creating a new one."
  mkdir "${BUILD_ANDROID_DIR}"
fi

declare -a args=(
  -G Ninja
  -B "${BUILD_ANDROID_DIR}"
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
  -DANDROID_ABI="${ANDROID_ABI}"
  -DANDROID_PLATFORM=android-29
  -DIREE_HOST_BINARY_ROOT="${IREE_HOST_BINARY_ROOT}"
  -DIREE_BUILD_COMPILER=OFF
  -DIREE_BUILD_TESTS=ON
  -DIREE_BUILD_SAMPLES=OFF
)

case "${BUILD_PRESET}" in
  test)
    args+=(
      -DIREE_ENABLE_ASSERTIONS=ON
    )
    ;;
  benchmark)
    args+=(
      -DIREE_ENABLE_ASSERTIONS=OFF
      -DIREE_BUILD_TESTS=OFF
    )
    ;;
  benchmark-with-tracing)
    args+=(
      -DIREE_ENABLE_ASSERTIONS=OFF
      -DIREE_BUILD_TESTS=OFF
      -DIREE_ENABLE_RUNTIME_TRACING=ON
    )
    ;;
  benchmark-suite-test)
    BUILD_BENCHMARK_SUITE_DIR="$(realpath ${BUILD_BENCHMARK_SUITE_DIR})"
    args+=(
      -DIREE_ENABLE_ASSERTIONS=ON
      -DIREE_BENCHMARK_SUITE_DIR="${BUILD_BENCHMARK_SUITE_DIR}"
    )
    ;;
  *)
    echo "Unknown build preset: ${BUILD_PRESET}"
    exit 1
    ;;
esac

# Configure towards 64-bit Android 10, then build.
"${CMAKE_BIN}" "${args[@]}"


echo "Building all for device"
echo "------------"
"${CMAKE_BIN}" --build "${BUILD_ANDROID_DIR}" -- -k 0

echo "Building test deps for device"
echo "------------------"
"${CMAKE_BIN}" --build "${BUILD_ANDROID_DIR}" --target iree-test-deps -- -k 0
