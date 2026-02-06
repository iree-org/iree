#! /usr/bin/env bash
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and test, using CMake/CTest, with Undefined Behavior Sanitizer
# instrumentation.
#
# See https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html. Components
# that don't work with UBSan are disabled.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_UBSAN_BUILD_DIR, defaulting to
# "build-ubsan". Designed for CI, but can be run manually. It reuses the build
# directory if it already exists. Expects to be run from the root of the IREE
# repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_UBSAN_BUILD_DIR:-build-ubsan}}"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"
IREE_REVERSE_ITERATION="${IREE_REVERSE_ITERATION:-OFF}"
# Enable CUDA and HIP/ROCM compiler and runtime by default if not on Darwin.
OFF_IF_DARWIN="$(uname | awk '{print ($1 == "Darwin") ? "OFF" : "ON"}')"
IREE_HAL_DRIVER_CUDA="${IREE_HAL_DRIVER_CUDA:-${OFF_IF_DARWIN}}"
IREE_HAL_DRIVER_HIP="${IREE_HAL_DRIVER_HIP:-${OFF_IF_DARWIN}}"
IREE_TARGET_BACKEND_CUDA="${IREE_TARGET_BACKEND_CUDA:-${OFF_IF_DARWIN}}"
IREE_TARGET_BACKEND_ROCM="${IREE_TARGET_BACKEND_ROCM:-${OFF_IF_DARWIN}}"

# Build hip tests for gfx942 to exercise codegen, but we won't run them.
# The user can override this to test other chips.
IREE_ROCM_TEST_TARGET_CHIP="${IREE_ROCM_TEST_TARGET_CHIP:-${IREE_HIP_TEST_TARGET_CHIP:-gfx942}}"

source build_tools/cmake/setup_build.sh

CMAKE_ARGS=(
  "-G" "Ninja"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
  "-DIREE_ENABLE_UBSAN=ON"
  "-B" "${BUILD_DIR?}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"

  # Python bindings are not compatible with sanitizers yet.
  "-DIREE_BUILD_PYTHON_BINDINGS=OFF"

  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"
  "-DIREE_REVERSE_ITERATION=${IREE_REVERSE_ITERATION}"
  "-DIREE_ENABLE_LLD=ON"
  "-DIREE_ENABLE_SPLIT_DWARF=ON"
  "-DIREE_ENABLE_THIN_ARCHIVES=ON"

  "-DIREE_HAL_DRIVER_CUDA=${IREE_HAL_DRIVER_CUDA}"
  "-DIREE_HAL_DRIVER_HIP=${IREE_HAL_DRIVER_HIP}"
  "-DIREE_TARGET_BACKEND_CUDA=${IREE_TARGET_BACKEND_CUDA}"
  "-DIREE_TARGET_BACKEND_ROCM=${IREE_TARGET_BACKEND_ROCM}"
  "-DIREE_ROCM_TEST_TARGET_CHIP=${IREE_ROCM_TEST_TARGET_CHIP}"

  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
)

echo "::group::Configuring CMake"
cmake "${CMAKE_ARGS[@]?}"
echo "::endgroup::"

echo "::group::Building all"
cmake --build "${BUILD_DIR?}" -- -k 0
echo "::endgroup::"

echo "::group::Building test deps"
cmake --build "${BUILD_DIR?}" --target iree-test-deps -- -k 0
echo "::endgroup::"

# Don't run any GPU tests for the time being.
# These are not ubsan-warning-free yet.
export IREE_VULKAN_DISABLE=1
export IREE_METAL_DISABLE=1
export IREE_CUDA_DISABLE=1
export IREE_HIP_DISABLE=1

echo "::group::Running tests"
build_tools/cmake/ctest_all.sh "${BUILD_DIR}"
echo "::endgroup::"

echo "::group::Running llvm-external-projects tests"
cmake --build "${BUILD_DIR?}" --target check-iree-dialects -- -k 0
echo "::endgroup::"
