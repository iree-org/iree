#!/bin/bash

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
# Build for the target (riscv64).
BUILD_RISCV_DIR="${BUILD_RISCV_DIR:-$ROOT_DIR/build-riscv}"
if [[ -d "${BUILD_RISCV_DIR?}" ]]; then
  echo "build-riscv directory already exists. Will use cached results there."
else
  echo "build-riscv directory does not already exist. Creating a new one."
  mkdir -p "${BUILD_RISCV_DIR?}"
fi

# Configure riscv, then build.
"${CMAKE_BIN?}" -G Ninja -B "${BUILD_RISCV_DIR?}" \
  -DCMAKE_TOOLCHAIN_FILE="$(realpath ${ROOT_DIR?}/build_tools/cmake/riscv.toolchain.cmake)" \
  -DIREE_HOST_BINARY_ROOT="$(realpath ${BUILD_HOST_DIR?}/install)" \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_ENABLE_MLIR=OFF \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_EMBEDDING_SAMPLES=ON \
  -DRISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT?}" \
  "${ROOT_DIR?}"
"${CMAKE_BIN?}" --build "${BUILD_RISCV_DIR?}"
