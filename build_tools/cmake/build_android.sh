#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the runtime using CMake targeting Android
#
# The required IREE_HOST_BIN_DIR environment variable indicates the location
# of the precompiled IREE binaries. Also requires that ANDROID_NDK variables be
# set. The BUILD_PRESET environment variable indicates how the project should be
# configured: "test", "benchmark", "benchmark-with-tracing", or
# "benchmark-suite-test". Defaults to "test".
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_TARGET_BUILD_DIR, defaulting to
# "build-android". Designed for CI, but can be run manually. It reuses the build
# directory if it already exists. Expects to be run from the root of the IREE
# repository.
#
# The default Android ABI is arm64-v8a, you can specify it with the variable
# IREE_ANDROID_ABI. See https://developer.android.com/ndk/guides/abis for the
# supported ABIs.


set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-android}}"
ANDROID_ABI="${IREE_ANDROID_ABI:-arm64-v8a}"
IREE_HOST_BIN_DIR="$(realpath ${IREE_HOST_BIN_DIR})"
E2E_TEST_ARTIFACTS_DIR="${E2E_TEST_ARTIFACTS_DIR:-build-e2e-test-artifacts/e2e_test_artifacts}"
BUILD_PRESET="${BUILD_PRESET:-test}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

declare -a args=(
  -G Ninja
  -B "${BUILD_DIR}"
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}"
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}"
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
  -DANDROID_ABI="${ANDROID_ABI}"
  -DANDROID_PLATFORM=android-29
  -DIREE_HOST_BIN_DIR="${IREE_HOST_BIN_DIR}"
  -DIREE_BUILD_COMPILER=OFF
  -DIREE_BUILD_TESTS=ON
  -DIREE_BUILD_ALL_CHECK_TEST_MODULES=OFF
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
    E2E_TEST_ARTIFACTS_DIR="$(realpath ${E2E_TEST_ARTIFACTS_DIR})"
    args+=(
      -DIREE_ENABLE_ASSERTIONS=ON
      -DIREE_E2E_TEST_ARTIFACTS_DIR="${E2E_TEST_ARTIFACTS_DIR}"
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

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi
