#!/bin/bash

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build "all" of the IREE project.
#
# Designed for CI, but can be run locally. The desired build directory can be
# passed as the first argument. Otherwise, it uses the environment variable
# IREE_BUILD_DIR, defaulting to "build". It reuses the build directory if it
# already exists. Expects to be run from the root of the IREE repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_BUILD_DIR:-build}}"
INSTALL_DIR="${IREE_INSTALL_DIR:-${BUILD_DIR}/install}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"
# Enable CUDA and HIP/ROCM compiler and runtime by default if not on Darwin.
# As with WebGPU the dependencies get fetched.
OFF_IF_DARWIN="$(uname | awk '{print ($1 == "Darwin") ? "OFF" : "ON"}')"
IREE_HAL_DRIVER_CUDA="${IREE_HAL_DRIVER_CUDA:-${OFF_IF_DARWIN}}"
IREE_HAL_DRIVER_HIP="${IREE_HAL_DRIVER_HIP:-${OFF_IF_DARWIN}}"
IREE_TARGET_BACKEND_CUDA="${IREE_TARGET_BACKEND_CUDA:-${OFF_IF_DARWIN}}"
IREE_TARGET_BACKEND_ROCM="${IREE_TARGET_BACKEND_ROCM:-${OFF_IF_DARWIN}}"
# Enable WebGPU compiler builds and tests by default. All deps get fetched as
# needed, but some of the deps are too large to enable by default for all
# developers.
IREE_TARGET_BACKEND_WEBGPU_SPIRV="${IREE_TARGET_BACKEND_WEBGPU_SPIRV:-${OFF_IF_DARWIN}}"
# Enable building the `iree-test-deps` target.
IREE_BUILD_TEST_DEPS="${IREE_BUILD_TEST_DEPS:-1}"
# If set to 1, exit after CMake configure (useful for generating compile_commands.json).
IREE_CONFIGURE_ONLY="${IREE_CONFIGURE_ONLY:-0}"

source build_tools/cmake/setup_build.sh
if (( ${IREE_USE_SCCACHE:-0} != 1 )); then
  source build_tools/cmake/setup_ccache.sh
fi

# Create install directory now--we need to get its real path later.
mkdir -p "${INSTALL_DIR}"

declare -a CMAKE_ARGS=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"

  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
  "-DCMAKE_INSTALL_PREFIX=$(realpath "${INSTALL_DIR}")"
  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"

  # Use `lld` for faster linking.
  "-DIREE_ENABLE_LLD=ON"

  # Enable split dwarf and thin archives for smaller object files and faster
  # linking (for builds with debug info).
  "-DIREE_ENABLE_SPLIT_DWARF=ON"
  "-DIREE_ENABLE_THIN_ARCHIVES=ON"

  # Enable docs build on the CI. The additional builds are pretty fast and
  # give us early warnings for some types of website publication errors.
  "-DIREE_BUILD_DOCS=ON"

  # Enable building the python bindings on CI.
  "-DIREE_BUILD_PYTHON_BINDINGS=ON"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DPYTHON_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"

  "-DIREE_HAL_DRIVER_CUDA=${IREE_HAL_DRIVER_CUDA}"
  "-DIREE_HAL_DRIVER_HIP=${IREE_HAL_DRIVER_HIP}"
  "-DIREE_TARGET_BACKEND_CUDA=${IREE_TARGET_BACKEND_CUDA}"
  "-DIREE_TARGET_BACKEND_ROCM=${IREE_TARGET_BACKEND_ROCM}"
  "-DIREE_TARGET_BACKEND_WEBGPU_SPIRV=${IREE_TARGET_BACKEND_WEBGPU_SPIRV}"
)

# Force /Z7 (embedded per-.obj debug info) instead of /Zi (shared per-target
# .pdb serialized through mspdbsrv.exe via /FS) on Windows.
#
# With /Zi, every cl.exe invocation in a target writes to the same
# LLVM*.pdb. The failure rate of this race scales with the number of
# concurrent writers, and MSVC 19.44.35227.0 broke it badly enough that
# LLVM-scale targets no longer survive: empirically on iree-org/iree CI,
# LLVMDemangle (6 TUs / shared PDB) loses ~3/6 to fatal error C1041
# ("cannot open program database"), and LLVMSupport (~165 TUs / shared
# PDB) loses ~165/165. The previous toolchain (19.44.35224.0) tolerated
# the same concurrency without failures.
#
# /Z7 eliminates the shared PDB entirely (debug info embedded in each
# .obj, consolidated by the linker into the final PDB), sidestepping the
# race regardless of toolchain. It is also the configuration sccache and
# ccache require, since they cannot cache outputs written to a shared
# external file.
if [[ "${OSTYPE}" =~ ^(msys|cygwin) ]]; then
  CMAKE_ARGS+=("-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded")
fi

"$CMAKE_BIN" "${CMAKE_ARGS[@]}"

if (( IREE_CONFIGURE_ONLY == 1 )); then
  echo "IREE_CONFIGURE_ONLY=1, exiting after configure"
  exit 0
fi

echo "Building all"
echo "------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" -- -k 0

echo "Building 'install'"
echo "------------------"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target install -- -k 0

if (( IREE_BUILD_TEST_DEPS == 1 )); then
  echo "Building test deps"
  echo "------------------"
  "$CMAKE_BIN" --build "${BUILD_DIR}" --target iree-test-deps -- -k 0
fi

if (( ${IREE_USE_SCCACHE:-0} == 1 )); then
  sccache --show-stats
elif (( ${IREE_USE_CCACHE:-0} == 1 )); then
  ccache --show-stats
fi
