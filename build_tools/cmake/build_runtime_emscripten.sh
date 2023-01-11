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

ROOT_DIR=$(git rev-parse --show-toplevel)
cd ${ROOT_DIR?}

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
IREE_HOST_BINARY_ROOT="$(realpath ${IREE_HOST_BINARY_ROOT})"

if [ -d "build-emscripten" ]
then
  echo "build-emscripten directory already exists. Will use cached results there."
else
  echo "build-emscripten directory does not already exist. Creating a new one."
  mkdir build-emscripten
fi
cd build-emscripten

# Configure using Emscripten's CMake wrapper, then build.
emcmake "${CMAKE_BIN?}" -G Ninja .. \
  -DIREE_HOST_BINARY_ROOT="${IREE_HOST_BINARY_ROOT}" \
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
