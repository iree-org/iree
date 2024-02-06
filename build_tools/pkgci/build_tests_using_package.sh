#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds test artifacts using a provided "package build".
#
# * The package directory to use is passed as the first argument.
# * Package builds can be obtained from the "iree-dist-*" native packages or
#   CMake "install" directories.
#
# Tests considered in-scope for this script:
#   * `runtime/` tests
#   * `tests/`, `tools/`, `samples/`, etc. tests from other directories that
#     use binaries from the CMake `IREE_HOST_BIN_DIR` option
#
# Tests considered out-of-scope for this script:
#   * `compiler/` tests and others using the `IREE_BUILD_COMPILER` CMake option

###############################################################################
# Script setup                                                                #
###############################################################################

set -xeuo pipefail


PACKAGE_DIR="$1"
BUILD_DIR="${BUILD_DIR:-build-tests}"

source build_tools/scripts/install_lit.sh

# CPU drivers and tests are enabled by default.
export IREE_CPU_DISABLE="${IREE_CPU_DISABLE:-0}"
# GPU drivers and tests are disabled by default.
export IREE_VULKAN_DISABLE="${IREE_VULKAN_DISABLE:-1}"
export IREE_METAL_DISABLE="${IREE_METAL_DISABLE:-1}"
export IREE_CUDA_DISABLE="${IREE_CUDA_DISABLE:-1}"

# Set cmake options based on disabled features.
declare -a cmake_config_options=()
if (( IREE_CPU_DISABLE == 1 )); then
  cmake_config_options+=("-DIREE_HAL_DRIVER_LOCAL_SYNC=OFF")
  cmake_config_options+=("-DIREE_HAL_DRIVER_LOCAL_TASK=OFF")
fi
if (( IREE_VULKAN_DISABLE == 1 )); then
  cmake_config_options+=("-DIREE_HAL_DRIVER_VULKAN=OFF")
fi
if (( IREE_METAL_DISABLE == 1 )); then
  cmake_config_options+=("-DIREE_HAL_DRIVER_METAL=OFF")
fi
if (( IREE_CUDA_DISABLE == 1 )); then
  cmake_config_options+=("-DIREE_HAL_DRIVER_CUDA=OFF")
fi

###############################################################################
# Build the runtime and compile 'test deps'                                   #
###############################################################################

echo "::group::Configure"
cmake_args=(
  "."
  "-G Ninja"
  "-B ${BUILD_DIR?}"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
  "-DIREE_BUILD_PYTHON_BINDINGS=OFF"
  "-DIREE_BUILD_COMPILER=OFF"
  "-DIREE_BUILD_ALL_CHECK_TEST_MODULES=OFF"
  "-DIREE_HOST_BIN_DIR=${PACKAGE_DIR?}/bin"
  "-DLLVM_EXTERNAL_LIT=${LLVM_EXTERNAL_LIT?}"
)
cmake_args+=(${cmake_config_options[@]})
cmake ${cmake_args[@]}
echo "::endgroup::"

echo "::group::Build runtime targets"
cmake --build ${BUILD_DIR?}
echo "::endgroup::"

echo "::group::Build iree-test-deps"
cmake --build ${BUILD_DIR?} --target iree-test-deps
echo "::endgroup::"
