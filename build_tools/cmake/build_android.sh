#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the runtime using CMake targeting Android
#
# The required IREE_HOST_BINARY_ROOT environment variable indicates the location
# of the precompiled IREE binaries. Also requires that ANDROID_ABI and
# ANDROID_NDK variables be set. The BUILD_PRESET environment variable indicates
# how the project should be configured: "test", "benchmark",
# "benchmark-with-tracing", or "benchmark-suite-test". Defaults to "test".
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_ANDROID_BUILD_DIR, defaulting to
# "build-android". Designed for CI, but can be run manually. It reuses the build
# directory if it already exists. Expects to be run from the root of the IREE
# repository.


set -xeuo pipefail

BUILD_DIR="${1:-${IREE_BUILD_ANDROID_DIR:-build-android}}"
IREE_HOST_BINARY_ROOT="$(realpath ${IREE_HOST_BINARY_ROOT})"
BUILD_BENCHMARK_SUITE_DIR="${BUILD_BENCHMARK_SUITE_DIR:-build-benchmarks/benchmark_suites}"
BUILD_PRESET="${BUILD_PRESET:-test}"

source build_tools/cmake/setup_build.sh

declare -a args=(
  -G Ninja
  -B "${BUILD_DIR}"
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
"${CMAKE_BIN}" --build "${BUILD_DIR}" -- -k 0

echo "Building test deps for device"
echo "------------------"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target iree-test-deps -- -k 0
