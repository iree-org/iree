#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the runtime using CMake targeting RISC-V
#
# The required IREE_HOST_BIN_DIR environment variable indicates the location
# of the precompiled IREE binaries.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_TARGET_BUILD_DIR, defaulting to
# "build-riscv". Designed for CI, but can be run manually. It reuses the build
# directory if it already exists. Expects to be run from the root of the IREE
# repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-riscv}}"
INSTALL_DIR="${IREE_INSTALL_DIR:-${BUILD_DIR}/install}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
CMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE:-$(realpath build_tools/cmake/linux_riscv64.cmake)}"
RISCV_TOOLCHAIN_PREFIX="${RISCV_TOOLCHAIN_PREFIX:-}"
IREE_HOST_BIN_DIR="$(realpath ${IREE_HOST_BIN_DIR})"
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
  "-DCMAKE_INSTALL_PREFIX=$(realpath ${INSTALL_DIR})"
  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"

  # Use `lld` for faster linking.
  "-DIREE_ENABLE_LLD=ON"

  # Cross compiling RISC-V
  "-DIREE_BUILD_ALL_CHECK_TEST_MODULES=OFF"
  "-DIREE_BUILD_COMPILER=OFF"
  "-DIREE_HOST_BIN_DIR=${IREE_HOST_BIN_DIR}"
  "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
  "-DRISCV_TOOLCHAIN_PREFIX=${RISCV_TOOLCHAIN_PREFIX}"
  "-DRISCV_TOOLCHAIN_ROOT=${RISCV_TOOLCHAIN_ROOT}"

  # Only enable RISC-V related drivers and targets.
  "-DIREE_HAL_DRIVER_DEFAULTS=OFF"
  "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON"
  "-DIREE_HAL_DRIVER_LOCAL_TASK=ON"
  "-DIREE_TARGET_BACKEND_DEFAULTS=OFF"
  "-DIREE_TARGET_BACKEND_LLVM_CPU=ON"
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

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi
