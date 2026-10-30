#!/bin/bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the runtime using CMake targeting MIPS
#
# The IREE_HOST_BIN_DIR environment variable indicates the location of the
# precompiled IREE binaries. If not set, defaults to "build/install/bin".
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_TARGET_BUILD_DIR, defaulting to
# "build-mips". Designed for CI, but can be run manually. It reuses the build
# directory if it already exists. Expects to be run from the root of the IREE
# repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-mips}}"
INSTALL_DIR="${IREE_INSTALL_DIR:-${BUILD_DIR}/install}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
CMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE:-$(realpath build_tools/cmake/linux_mips64_gcc.cmake)}"
MIPS_TOOLCHAIN_PREFIX="${MIPS_TOOLCHAIN_PREFIX:-mips64el-linux-gnuabi64-}"
MIPS_TOOLCHAIN_ROOT="${MIPS_TOOLCHAIN_ROOT:-}"
IREE_HOST_BIN_DIR="$(realpath "${IREE_HOST_BIN_DIR:-build/install/bin}")"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"
# Enable building the `iree-test-deps` target.
IREE_BUILD_TEST_DEPS="${IREE_BUILD_TEST_DEPS:-1}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

# Create install directory now--we need to get its real path later.
mkdir -p "${INSTALL_DIR}"

declare -a args
args=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"

  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DCMAKE_INSTALL_PREFIX=$(realpath "${INSTALL_DIR}")"
  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"

  # Use GNU linker, not LLD
  "-DIREE_ENABLE_LLD=OFF"

  # Cross compiling MIPS
  "-DIREE_BUILD_ALL_CHECK_TEST_MODULES=OFF"
  "-DIREE_BUILD_COMPILER=OFF"
  "-DIREE_BUILD_SAMPLES=OFF"
  "-DIREE_HOST_BIN_DIR=${IREE_HOST_BIN_DIR}"
  "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
  "-DMIPS_TOOLCHAIN_PREFIX=${MIPS_TOOLCHAIN_PREFIX}"
  "-DMIPS_TOOLCHAIN_ROOT=${MIPS_TOOLCHAIN_ROOT}"

  # VMVX only
  "-DIREE_HAL_DRIVER_DEFAULTS=OFF"
  "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON"
  "-DIREE_HAL_DRIVER_LOCAL_TASK=ON"
  "-DIREE_TARGET_BACKEND_DEFAULTS=OFF"
  "-DIREE_TARGET_BACKEND_LLVM_CPU=OFF" # off (vmvx only)


)

"${CMAKE_BIN}" "${args[@]}"
echo "Building all"
echo "------------"
"${CMAKE_BIN}" --build "${BUILD_DIR}" -- -k 0

echo "Building 'install'"
echo "------------------"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target install -- -k 0

if (( IREE_BUILD_TEST_DEPS == 1 )); then
  echo "Building test deps"
  echo "------------------"
  "${CMAKE_BIN}" --build "${BUILD_DIR}" --target iree-test-deps -- -k 0
fi
