#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and test, using CMake/CTest, with AddressSanitizer instrumentation.
#
# See https://clang.llvm.org/docs/AddressSanitizer.html. Components that don't
# work with ASan (e.g. Python bindings) are disabled. Some tests that fail under
# ASan are individually excluded.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_ASAN_BUILD_DIR, defaulting to "build-asan".
# Designed for CI, but can be run manually. It reuses the build directory if it
# already exists. Expects to be run from the root of the IREE repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_ASAN_BUILD_DIR:-build-asan}}"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

CMAKE_ARGS=(
  "-G" "Ninja"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
  "-DIREE_ENABLE_ASAN=ON"
  "-B" "${BUILD_DIR?}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DPYTHON_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"

  # Also check if microbenchmarks are buildable.
  "-DIREE_BUILD_MICROBENCHMARKS=ON"

  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"
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

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi

# Respect the user setting, but default to as many jobs as we have cores.
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

# Respect the user setting, but default to turning on Vulkan.
export IREE_VULKAN_DISABLE=${IREE_VULKAN_DISABLE:-0}
# Respect the user setting, but default to turning off CUDA.
export IREE_CUDA_DISABLE=${IREE_CUDA_DISABLE:-1}
# The VK_KHR_shader_float16_int8 extension is optional prior to Vulkan 1.2.
# We test on SwiftShader as a baseline, which does not support this extension.
export IREE_VULKAN_F16_DISABLE=${IREE_VULKAN_F16_DISABLE:-1}
# Respect the user setting, but default to skipping tests that require Nvidia GPU.
export IREE_NVIDIA_GPU_TESTS_DISABLE=${IREE_NVIDIA_GPU_TESTS_DISABLE:-1}

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
if [[ "${IREE_NVIDIA_GPU_TESTS_DISABLE}" == 1 ]]; then
  label_exclude_args+=("^requires-gpu-nvidia$")
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
