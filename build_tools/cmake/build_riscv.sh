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
BUILD_TYPE="${IREE_BUILD_TYPE:-RelWithDebInfo}"
RISCV_PLATFORM="${IREE_TARGET_PLATFORM:-linux}"
RISCV_ARCH="${IREE_TARGET_ARCH:-riscv_64}"
RISCV_COMPILER_FLAGS="${RISCV_COMPILER_FLAGS:--O3}"
IREE_HOST_BIN_DIR="$(realpath ${IREE_HOST_BIN_DIR})"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

RISCV_PLATFORM_ARCH="${RISCV_PLATFORM}-${RISCV_ARCH}"
echo "Build riscv target with the config of ${RISCV_PLATFORM_ARCH}"
TOOLCHAIN_FILE="$(realpath build_tools/cmake/riscv.toolchain.cmake)"
declare -a args
args=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DPYTHON_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}"
  "-DIREE_HOST_BIN_DIR=${IREE_HOST_BIN_DIR}"
  "-DRISCV_CPU=${RISCV_PLATFORM_ARCH}"
  "-DRISCV_COMPILER_FLAGS=${RISCV_COMPILER_FLAGS}"
  "-DIREE_ENABLE_ASSERTIONS=ON"
  "-DIREE_BUILD_SAMPLES=ON"
  "-DIREE_BUILD_COMPILER=OFF"
  "-DIREE_BUILD_ALL_CHECK_TEST_MODULES=OFF"
  # CPU info doesn't work on RISCV
  "-DIREE_ENABLE_CPUINFO=OFF"
)

if [[ "${RISCV_PLATFORM}" == "linux" ]]; then
  args+=(
    -DRISCV_TOOLCHAIN_ROOT="${RISCV_RV64_LINUX_TOOLCHAIN_ROOT}"
    -DRISCV_TOOLCHAIN_PREFIX="riscv64-unknown-linux-gnu-"
  )
elif [[ "${RISCV_PLATFORM_ARCH}" == "generic-riscv_32" ]]; then
  args+=(
    # TODO(#6353): Off until tools/ are refactored to support threadless config.
    -DIREE_BUILD_TESTS=OFF
    -DRISCV_TOOLCHAIN_ROOT="${RISCV_RV32_NEWLIB_TOOLCHAIN_ROOT}"
    -DRISCV_TOOLCHAIN_PREFIX="riscv32-unknown-elf"
  )
else
  echo "riscv config for ${RISCV_PLATFORM_ARCH} not supported yet"
  return -1
fi

"${CMAKE_BIN}" "${args[@]}"

"${CMAKE_BIN}" --build "${BUILD_DIR}" -- -k 0
if [[ "${RISCV_PLATFORM}" == "linux" ]]; then
  echo "Building test deps for RISC-V"
  echo "-----------------------------"
  "${CMAKE_BIN}" --build "${BUILD_DIR}" --target iree-test-deps -- -k 0
fi
