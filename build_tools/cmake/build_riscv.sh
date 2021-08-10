#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the IREE project towards RISCV with CMake. Designed for CI,
# but can be run manually. This uses previously cached build results and does
# not clear build directories.
#
# Host binaries (e.g. compiler tools) will be built and installed in build-host/
# RISCV binaries (e.g. tests) will be built in build-riscv/.

set -x
set -e

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
CMAKE_BIN="${CMAKE_BIN:-$(which cmake)}"
RISCV_CONFIG="${RISCV_CONFIG:-rv64}"

"${CMAKE_BIN?}" --version
ninja --version

cd "${ROOT_DIR?}"

# --------------------------------------------------------------------------- #
# Build for the host.
BUILD_HOST_DIR="${BUILD_HOST_DIR:-$ROOT_DIR/build-host}"
if [[ -d "${BUILD_HOST_DIR?}" ]]; then
  echo "build-host directory already exists. Will use cached results there."
else
  echo "build-host directory does not already exist. Creating a new one."
  mkdir -p "${BUILD_HOST_DIR?}"
fi

# Configure, build, install.
"${CMAKE_BIN?}" -G Ninja -B "${BUILD_HOST_DIR?}" \
  -DCMAKE_INSTALL_PREFIX="${BUILD_HOST_DIR?}/install" \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  "${ROOT_DIR?}"
"${CMAKE_BIN?}" --build "${BUILD_HOST_DIR?}" --target install
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Build for the target (riscv).
BUILD_RISCV_DIR="${BUILD_RISCV_DIR:-$ROOT_DIR/build-riscv}"
if [[ -d "${BUILD_RISCV_DIR?}" ]]; then
  echo "build-riscv directory already exists. Will use cached results there."
else
  echo "build-riscv directory does not already exist. Creating a new one."
  mkdir -p "${BUILD_RISCV_DIR?}"
fi

echo "Build riscv target with the config of ${RISCV_CONFIG?}"
declare -a args
args=(
  "-G" "Ninja"
  "-B" "${BUILD_RISCV_DIR?}"
  -DCMAKE_TOOLCHAIN_FILE="$(realpath ${ROOT_DIR?}/build_tools/cmake/riscv.toolchain.cmake)"
  -DIREE_HOST_BINARY_ROOT="$(realpath ${BUILD_HOST_DIR?}/install)"
  -DRISCV_CPU="${RISCV_CONFIG?}"
  -DIREE_BUILD_COMPILER=OFF
  -DIREE_BUILD_SAMPLES=ON
)

if [[ "${RISCV_CONFIG?}" == "rv64" ]]; then
  args+=(
    -DRISCV_TOOLCHAIN_ROOT="${RISCV_RV64_LINUX_TOOLCHAIN_ROOT?}"
  )
elif [[ "${RISCV_CONFIG?}" == "rv32-baremetal" ]]; then
  args+=(
    # TODO(#6353): Off until iree/tools are refactored to support threadless config.
    -DIREE_BUILD_TESTS=OFF
    -DRISCV_TOOLCHAIN_ROOT="${RISCV_RV32_NEWLIB_TOOLCHAIN_ROOT?}"
  )
else
  echo "riscv config not supported yet"
  return -1
fi

args_str=$(IFS=' ' ; echo "${args[*]}")
"${CMAKE_BIN?}" ${args_str} "${ROOT_DIR?}"
"${CMAKE_BIN?}" --build "${BUILD_RISCV_DIR?}"
