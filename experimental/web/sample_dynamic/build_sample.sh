#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds the sample, running host tools, Emscripten, and CMake as needed.
#
# Prerequisites:
#   * Environment must be configured for Emscripten
#   * Host tools must be built (default at IREE_SOURCE_DIR/build-host/install).
#     The build_tools/cmake/build_host_install.sh script can do this for you.
#
# Usage:
#   build_sample.sh (optional install path) && serve_sample.sh

set -e

###############################################################################
# Setup and checking for dependencies                                         #
###############################################################################

if ! command -v emcmake &> /dev/null
then
  echo "'emcmake' not found, setup environment according to https://emscripten.org/docs/getting_started/downloads.html"
  exit 1
fi

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
ROOT_DIR=$(git rev-parse --show-toplevel)
SOURCE_DIR=${ROOT_DIR}/experimental/web/sample_dynamic

BUILD_DIR=${ROOT_DIR?}/build-emscripten
mkdir -p ${BUILD_DIR}

BINARY_DIR=${BUILD_DIR}/experimental/web/sample_dynamic
mkdir -p ${BINARY_DIR}

INSTALL_ROOT="${1:-${ROOT_DIR}/build-host/install}"

###############################################################################
# Compile from .mlir input to portable .vmfb file using host tools            #
###############################################################################

COMPILE_TOOL="${INSTALL_ROOT?}/bin/iree-compile"

compile_sample() {
  echo "  Compiling '$1' sample..."
  ${COMPILE_TOOL?} $2 \
    --iree-mlir-to-vm-bytecode-module \
    --iree-input-type=mhlo \
    --iree-hal-target-backends=llvm \
    --iree-llvm-target-triple=wasm32-unknown-emscripten \
    --iree-llvm-target-cpu-features=+atomics,+bulk-memory,+simd128 \
    --o ${BINARY_DIR}/$1.vmfb
}

echo "=== Compiling sample MLIR files to VM flatbuffer outputs (.vmfb) ==="
compile_sample "simple_abs"     "${ROOT_DIR?}/iree/samples/models/simple_abs.mlir"
compile_sample "fullyconnected" "${ROOT_DIR?}/iree/test/e2e/models/fullyconnected.mlir"
compile_sample "collatz"        "${ROOT_DIR?}/iree/test/e2e/models/collatz.mlir"

###############################################################################
# Build the web artifacts using Emscripten                                    #
###############################################################################

echo "=== Building web artifacts using Emscripten ==="

pushd ${BUILD_DIR}

# Configure using Emscripten's CMake wrapper, then build.
# Note: The sample creates a device directly, so no drivers are required.
emcmake "${CMAKE_BIN?}" -G Ninja .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_HOST_BINARY_ROOT=${INSTALL_ROOT} \
  -DIREE_BUILD_EXPERIMENTAL_WEB_SAMPLES=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF

"${CMAKE_BIN?}" --build . --target \
  iree_experimental_web_sample_dynamic_sync

popd

echo "=== Copying static files (.html, .js) to the build directory ==="

cp ${SOURCE_DIR?}/index.html ${BINARY_DIR}
cp ${SOURCE_DIR?}/iree_api.js ${BINARY_DIR}
cp ${SOURCE_DIR?}/iree_worker.js ${BINARY_DIR}
