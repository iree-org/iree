#!/bin/bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the runtime and the samples/baremetal_riscv64 runners for
# bare-metal RISC-V 64 using build_tools/cmake/generic_riscv64_gcc.cmake.
#
# Requires RISCV_TOOLCHAIN_ROOT to point to a riscv-none-elf GCC toolchain
# whose newlib was built with -mcmodel=medany and that provides
# semihost.specs, such as xPack riscv-none-elf-gcc.
#
# IREE_HOST_BIN_DIR specifies the directory containing the prebuilt IREE host
# tools and defaults to "build/install/bin".
#
# Pass the desired build directory as the first argument. Otherwise,
# IREE_TARGET_BUILD_DIR is used and defaults to "build-riscv-baremetal".
#
# This script is designed for CI but can also be run manually. It must be run
# from the IREE repository root.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-riscv-baremetal}}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"
IREE_HOST_BIN_DIR="$(realpath "${IREE_HOST_BIN_DIR:-build/install/bin}")"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

declare -a args
args=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"

  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"

  # Cross compiling bare-metal RISC-V.
  "-DCMAKE_TOOLCHAIN_FILE=$(realpath build_tools/cmake/generic_riscv64_gcc.cmake)"
  "-DRISCV_TOOLCHAIN_ROOT=${RISCV_TOOLCHAIN_ROOT}"
  "-DIREE_BUILD_COMPILER=OFF"
  "-DIREE_BUILD_SAMPLES=ON"
  "-DIREE_HOST_BIN_DIR=${IREE_HOST_BIN_DIR}"

  # No HAL device drivers: the sample uses the inline HAL, which runs the
  # module without creating a HAL device. The embedded-ELF loader (used by
  # the hal_loader module for llvm-cpu kernels) must be enabled explicitly
  # because loaders default off when no local driver is enabled.
  "-DIREE_HAL_DRIVER_DEFAULTS=OFF"
  "-DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON"
)

"${CMAKE_BIN}" "${args[@]}"
# Build only the bare-metal sample and the runtime targets it depends on.
# Other sample targets may not support this GCC/newlib configuration.
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target samples/baremetal_riscv64/all -- -k 0

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi
