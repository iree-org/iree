#!/bin/bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the runtime + test deps for MIPS64 Release 6 with MSA (128-bit
# SIMD), i.e. the native llvm-cpu codegen configuration (vs build_mips.sh, which
# is the scalar r2 / VMVX runtime). Uses the distro-packaged r6 toolchain
# (prefix mipsisa64r6el-linux-gnuabi64-); install it with:
#   sudo apt install gcc-mipsisa64r6el-linux-gnuabi64 \
#     g++-mipsisa64r6el-linux-gnuabi64 libc6-dev-mips64r6el-cross qemu-user
# That toolchain ships the mips64r6 / n64 / little-endian / NaN-2008 compiler +
# sysroot the generic mips64el-linux-gnuabi64 (r2) toolchain lacks.
#
# The IREE_HOST_BIN_DIR environment variable indicates the location of the
# precompiled IREE host binaries (the compiler that emits the MIPS .so). If not
# set, defaults to "build/install/bin".
#
# The desired build directory can be passed as the first argument. Otherwise it
# uses IREE_TARGET_BUILD_DIR, defaulting to "build-mips-r6". Designed for CI, can
# be run manually. Reuses the build dir if it exists. Run from the repo root.
#
# NOTE: the e2e test modules are compiled by the host iree-compile. Ninja tracks
# it by path, not the libIREECompiler.so it dlopens, so after rebuilding the
# host compiler the modules are stale; a clean checkout / CI is unaffected, but
# a local re-run after a compiler change needs:
#   find "${BUILD_DIR}" -name '*.vmfb' -delete
# before rebuilding.

# No `set -x`: it echoes every command (each configure flag, every assignment)
# and doubles the "Building ..." progress markers below. Those echoes mark
# progress on their own, and ninja surfaces any real build error itself.
set -euo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-mips-r6}}"
INSTALL_DIR="${IREE_INSTALL_DIR:-${BUILD_DIR}/install}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
CMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE:-$(realpath build_tools/cmake/linux_mips64r6_gcc.cmake)}"
IREE_HOST_BIN_DIR="$(realpath "${IREE_HOST_BIN_DIR:-build/install/bin}")"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"
# Enable building the `iree-test-deps` target.
IREE_BUILD_TEST_DEPS="${IREE_BUILD_TEST_DEPS:-1}"

# The apt r6 toolchain (mipsisa64r6el-linux-gnuabi64-*) is on PATH and its
# sysroot (/usr/mipsisa64r6el-linux-gnuabi64, from libc6-dev-mips64r6el-cross)
# already ships the C++/gcc runtime libs (libstdc++.so.6, libgcc_s, libatomic),
# so no toolchain root or sysroot library-copy step is needed here.

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

  # Cross compiling MIPS r6 + MSA
  "-DIREE_BUILD_COMPILER=OFF"
  # Build samples: samples/simple_embedding has a mips_64 case and its
  # embedded native_test runs under QEMU as an end-to-end embedded-ELF check.
  # Under this config (compiler off, CPU-only drivers) it is the only sample
  # that builds a cross target; the rest self-skip or are Python/MLIR-only.
  "-DIREE_BUILD_SAMPLES=ON"
  "-DIREE_HOST_BIN_DIR=${IREE_HOST_BIN_DIR}"
  "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"

  # Only build check-test modules that are actually tested here. With the GPU
  # backends' drivers disabled below, their fixtures would otherwise be built
  # (and fail) by the host compiler; OFF keeps the build clean.
  "-DIREE_BUILD_ALL_CHECK_TEST_MODULES=OFF"

  # The r6/MSA gate defaults to the embedded-ELF loader (link_embedded=true,
  # MIPS relocations in hal/local/elf/arch/mips_64.c), which is already on by
  # default. Keep the system dynamic library loader on too so the
  # link_embedded=false path still works (e.g. compiling a model with
  # --iree-llvmcpu-link-embedded=false for a system .so).
  "-DIREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY=ON"

  # local CPU drivers only (no GPU); target backends come from the host compiler
  "-DIREE_HAL_DRIVER_DEFAULTS=OFF"
  "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON"
  "-DIREE_HAL_DRIVER_LOCAL_TASK=ON"
  "-DIREE_TARGET_BACKEND_DEFAULTS=OFF"
  "-DIREE_TARGET_BACKEND_LLVM_CPU=OFF"
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
