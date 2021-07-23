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
# Host binaries (e.g. compiler tools) will be built and installed in build-host/
# Emscripten binaries (e.g. .wasm and .js files) will be built in
# build-emscripten/.

set -x
set -e

if ! command -v emcmake &> /dev/null
then
    echo "'emcmake' not found, setup environment according to https://emscripten.org/docs/getting_started/downloads.html"
    exit
fi

ROOT_DIR=$(git rev-parse --show-toplevel)

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}

"${CMAKE_BIN?}" --version
ninja --version

cd ${ROOT_DIR?}

# --------------------------------------------------------------------------- #
# Build for the host.

if [ -d "build-host" ]
then
  echo "build-host directory already exists. Will use cached results there."
else
  echo "build-host directory does not already exist. Creating a new one."
  mkdir build-host
fi
cd build-host

# Configure, build, install.
# Just build the host components that we need for an Emscripten runtime build.
"${CMAKE_BIN?}" -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_TARGET_BACKENDS_TO_BUILD=VMVX\;WASM-LLVM-AOT \
  -DIREE_HAL_DRIVERS_TO_BUILD= \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF
"${CMAKE_BIN?}" --build . --target install
# --------------------------------------------------------------------------- #

cd ${ROOT_DIR?}

# --------------------------------------------------------------------------- #
# Build for the target (Emscripten).

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
  -DIREE_HOST_BINARY_ROOT=$PWD/../build-host/install \
  -DIREE_HAL_DRIVERS_TO_BUILD=VMVX\;DyLib \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_ENABLE_MLIR=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=ON

# TODO(scotttodd): expand this list of targets
"${CMAKE_BIN?}" --build . --target iree_samples_simple_embedding_simple_embedding_vmvx_sync
