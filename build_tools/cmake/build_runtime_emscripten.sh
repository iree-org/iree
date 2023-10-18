#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile IREE's runtime through Emscripten to WebAssembly with CMake.
# Designed for CI, but can be run manually. This uses previously cached build
# results and does not clear build directories.
#
# Host binaries (e.g. compiler tools) should already be built at
# ./build-host/install. Emscripten binaries (e.g. .wasm and .js files) will be
# built in ./build-emscripten/.

set -xeuo pipefail

if ! command -v emcmake &> /dev/null
then
    echo "'emcmake' not found, setup environment according to https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

BUILD_DIR="build-emscripten"
IREE_HOST_BIN_DIR="$(realpath ${IREE_HOST_BIN_DIR})"

source build_tools/cmake/setup_build.sh
# Note: not using ccache since the runtime build should be fast already.

cd "${BUILD_DIR}"

# Configure using Emscripten's CMake wrapper, then build.
emcmake "${CMAKE_BIN?}" -G Ninja .. \
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DIREE_HOST_BIN_DIR="${IREE_HOST_BIN_DIR}" \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_ENABLE_CPUINFO=OFF \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_BUILD_SAMPLES=ON

echo "Building default targets"
echo "------------------------"
"${CMAKE_BIN?}" --build . -- -k 0

echo "Building test deps"
echo "------------------"
"${CMAKE_BIN?}" --build . --target iree-test-deps -- -k 0
