#!/bin/bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the runtime + test deps for MIPS64 Release 6 with MSA (128-bit
# SIMD), i.e. the native llvm-cpu codegen configuration (vs build_mips.sh, which
# is the scalar r2 / VMVX runtime). Requires the Codescape IMG GNU toolchain
# (prefix mips-img-linux-gnu-), whose root is passed via MIPS_TOOLCHAIN_ROOT;
# that toolchain ships the mips64r6 / n64 / little-endian / NaN-2008 multilib
# the distro mips64el-linux-gnuabi64 toolchain lacks.
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

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-mips-r6}}"
INSTALL_DIR="${IREE_INSTALL_DIR:-${BUILD_DIR}/install}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
CMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE:-$(realpath build_tools/cmake/linux_mips64r6_gcc.cmake)}"
MIPS_TOOLCHAIN_PREFIX="${MIPS_TOOLCHAIN_PREFIX:-mips-img-linux-gnu-}"
MIPS_TOOLCHAIN_ROOT="${MIPS_TOOLCHAIN_ROOT:-}"
# Per-multilib sysroot that matches the r6 / n64 / little-endian flags.
MIPS_R6_SYSROOT_VARIANT="${MIPS_R6_SYSROOT_VARIANT:-mipsel-r6-hard}"
IREE_HOST_BIN_DIR="$(realpath "${IREE_HOST_BIN_DIR:-build/install/bin}")"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"
# Enable building the `iree-test-deps` target.
IREE_BUILD_TEST_DEPS="${IREE_BUILD_TEST_DEPS:-1}"

if [[ -z "${MIPS_TOOLCHAIN_ROOT}" ]]; then
  echo "error: MIPS_TOOLCHAIN_ROOT must point to the extracted Codescape IMG" >&2
  echo "       toolchain root (contains bin/mips-img-linux-gnu-gcc and sysroot/)," >&2
  echo "       e.g. \$HOME/toolchains/mips-img-linux-gnu/2024.10-03" >&2
  exit 1
fi

# The Codescape sysroot carries libc but not the C++/gcc runtime libs (those
# live in the compiler tree). QEMU's loader resolves them from the sysroot, so
# copy them in once (idempotent) for the C++ test binaries (iree-check-module,
# the cc_tests). Harmless if already present.
MIPS_SYSROOT="${MIPS_TOOLCHAIN_ROOT}/sysroot/${MIPS_R6_SYSROOT_VARIANT}"
MIPS_RT_LIBDIR="${MIPS_TOOLCHAIN_ROOT}/${MIPS_TOOLCHAIN_PREFIX%-}/lib/${MIPS_R6_SYSROOT_VARIANT}/lib64"
if [[ -d "${MIPS_SYSROOT}/lib64" && -d "${MIPS_RT_LIBDIR}" ]]; then
  cp -a -n "${MIPS_RT_LIBDIR}"/libstdc++.so.6* \
           "${MIPS_RT_LIBDIR}"/libgcc_s.so.1 \
           "${MIPS_RT_LIBDIR}"/libatomic.so* \
           "${MIPS_SYSROOT}/lib64/" 2>/dev/null || true
fi

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
  "-DIREE_BUILD_SAMPLES=OFF"
  "-DIREE_HOST_BIN_DIR=${IREE_HOST_BIN_DIR}"
  "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
  "-DMIPS_TOOLCHAIN_PREFIX=${MIPS_TOOLCHAIN_PREFIX}"
  "-DMIPS_TOOLCHAIN_ROOT=${MIPS_TOOLCHAIN_ROOT}"

  # Only build check-test modules that are actually tested here. With the GPU
  # backends' drivers disabled below, their fixtures would otherwise be built
  # (and fail) by the host compiler; OFF keeps the build clean.
  "-DIREE_BUILD_ALL_CHECK_TEST_MODULES=OFF"

  # Native codegen runs the compiled MIPS .so via the system dynamic library
  # loader (link_embedded=false path), so that loader must be enabled.
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
