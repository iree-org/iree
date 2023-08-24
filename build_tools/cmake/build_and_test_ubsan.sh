#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and test, using CMake/CTest, with UndefinedBehaviorSanitizer
# instrumentation.
#
# See https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_UBSAN_BUILD_DIR, defaulting to
# "build-ubsan". Designed for CI, but can be run manually. It reuses the build
# directory if it already exists. Expects to be run from the root of the IREE
# repository.

set -euo pipefail

BUILD_DIR="${1:-${IREE_UBSAN_BUILD_DIR:-build-ubsan}}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

CMAKE_ARGS=(
  "-G" "Ninja"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DPYTHON_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
  "-DIREE_ENABLE_LLD=ON"
  "-DIREE_ENABLE_UBSAN=ON"
)

"${CMAKE_BIN}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]?}"

echo "Building all"
echo "------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" -- -k 0

echo "Building test deps"
echo "------------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" --target iree-test-deps -- -k 0

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi

# Respect the user setting, but default to as many jobs as we have cores.
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

# Respect the user setting, but default to turning on Vulkan.
export IREE_VULKAN_DISABLE="${IREE_VULKAN_DISABLE:-0}"
# Respect the user setting, but default to turning off Metal.
export IREE_METAL_DISABLE="${IREE_METAL_DISABLE:-1}"
# Respect the user setting, but default to turning off CUDA.
export IREE_CUDA_DISABLE="${IREE_CUDA_DISABLE:-1}"

# Honor the "noubsan" label on tests.
export IREE_EXTRA_COMMA_SEPARATED_CTEST_LABELS_TO_EXCLUDE=noubsan

# Run all tests.
build_tools/cmake/ctest_all.sh "${BUILD_DIR}"
