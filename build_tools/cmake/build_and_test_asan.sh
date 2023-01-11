#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Note: this script diverges from the non-ASan build in a few ways:
#   * The CMake build sets `IREE_ENABLE_ASAN=ON`
#   * Omit optional components that don't work with ASan (e.g. Python bindings)
#   * Some tests that fail under ASan are individually excluded
#
# The desired build directory can be passed as
# the first argument. Otherwise, it uses the environment variable
# IREE_ASAN_BUILD_DIR, defaulting to "build-asan". Designed for CI, but
# can be run manually. This reuses the build directory if it already exists.
#
# Build and test the project with CMake with ASan enabled and using
# SwiftShader's software Vulkan driver.
# ASan docs: https://clang.llvm.org/docs/AddressSanitizer.html


set -xeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}"

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
BUILD_DIR="${1:-${IREE_ASAN_BUILD_DIR:-build-asan}}"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"

"$CMAKE_BIN" --version
ninja --version

if [[ -d "${BUILD_DIR}" ]]; then
  echo "Build directory '${BUILD_DIR}' already exists. Will use cached results there."
else
  echo "Build directory '${BUILD_DIR}' does not already exist. Creating a new one."
  mkdir "${BUILD_DIR}"
fi

CMAKE_ARGS=(
  "-G" "Ninja"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
  "-DIREE_ENABLE_ASAN=ON"
  "-B" "${BUILD_DIR?}"

  # Also check if microbenchmarks are buildable.
  "-DIREE_BUILD_MICROBENCHMARKS=ON"

  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"

  # Enable CUDA compiler and runtime builds unconditionally. Our CI images all
  # have enough deps to at least build CUDA support and compile CUDA binaries
  # (but not necessarily test on real hardware).
  "-DIREE_HAL_DRIVER_CUDA=ON"
  "-DIREE_TARGET_BACKEND_CUDA=ON"
)

echo "Configuring CMake"
"${CMAKE_BIN?}" "${CMAKE_ARGS[@]?}"

echo "Building all"
echo "------------"
"${CMAKE_BIN?}" --build "${BUILD_DIR?}" -- -k 0

echo "Building test deps"
echo "------------------"
"${CMAKE_BIN?}" --build "${BUILD_DIR?}" --target iree-test-deps -- -k 0

echo "Building microbenchmark suites"
echo "------------------"
"${CMAKE_BIN?}" --build "${BUILD_DIR?}" --target iree-microbenchmark-suites -- -k 0

# Respect the user setting, but default to as many jobs as we have cores.
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

# Respect the user setting, but default to turning on Vulkan.
export IREE_VULKAN_DISABLE=${IREE_VULKAN_DISABLE:-0}
# Respect the user setting, but default to turning off CUDA.
export IREE_CUDA_DISABLE=${IREE_CUDA_DISABLE:-1}
# The VK_KHR_shader_float16_int8 extension is optional prior to Vulkan 1.2.
# We test on SwiftShader as a baseline, which does not support this extension.
export IREE_VULKAN_F16_DISABLE=${IREE_VULKAN_F16_DISABLE:-1}

# Tests to exclude by label. In addition to any custom labels (which are carried
# over from Bazel tags), every test should be labeled with its directory.
declare -a label_exclude_args=(
  # Exclude specific labels.
  # Put the whole label with anchors for exact matches.
  # For example:
  #   ^nokokoro$
  ^nokokoro$

  # Exclude all tests in a directory.
  # Put the whole directory with anchors for exact matches.
  # For example:
  #   ^bindings/python/iree/runtime$

  # Exclude all tests in some subdirectories.
  # Put the whole parent directory with only a starting anchor.
  # Use a trailing slash to avoid prefix collisions.
  # For example:
  #   ^bindings/
)

# IREE_VULKAN_DISABLE is handled separately as we run Vulkan and non-Vulkan
# tests in separate ctest commands anyway.
if [[ "${IREE_CUDA_DISABLE?}" == 1 ]]; then
  label_exclude_args+=("^driver=cuda$")
fi

if [[ "${IREE_VULKAN_F16_DISABLE?}" == 1 ]]; then
  label_exclude_args+=("^vulkan_uses_vk_khr_shader_float16_int8$")
fi

label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]?}"))"

vulkan_label_regex='^driver=vulkan$'

cd ${BUILD_DIR?}

echo "*** Running main project ctests that do not use the Vulkan driver *******"
ctest \
  --timeout 900 \
  --output-on-failure \
  --no-tests=error \
  --label-exclude "${label_exclude_regex}|${vulkan_label_regex}"

echo "******************** llvm-external-projects tests ***********************"
cmake --build . --target check-iree-dialects -- -k 0

if [[ "${IREE_VULKAN_DISABLE?}" == 0 ]]; then
  echo "*** Running ctests that use the Vulkan driver, with LSAN disabled *****"
  # Disable LeakSanitizer (LSAN) because of a history of issues with Swiftshader
  # (#5716, #8489, #11203).
  ASAN_OPTIONS=detect_leaks=0 \
    ctest \
      --timeout 900 \
      --output-on-failure \
      --no-tests=error \
      --label-regex "${vulkan_label_regex}" \
      --label-exclude "${label_exclude_regex}"
fi
