#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -e

###############################################################################
# Setup and checking for dependencies                                         #
###############################################################################

if ! command -v emcmake &> /dev/null
then
  echo "'emcmake' not found, setup environment according to https://emscripten.org/docs/getting_started/downloads.html"
  exit
fi

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
ROOT_DIR=$(git rev-parse --show-toplevel)
SOURCE_DIR=${ROOT_DIR}/experimental/web/sample_dynamic

BUILD_DIR=${ROOT_DIR?}/build-emscripten
mkdir -p ${BUILD_DIR}

BINARY_DIR=${BUILD_DIR}/experimental/web/sample_dynamic
mkdir -p ${BINARY_DIR}

###############################################################################
# Compile from .mlir input to portable .vmfb file using host tools            #
###############################################################################

# TODO(scotttodd): portable path ... discover from python install if on $PATH?
INSTALL_ROOT="D:\dev\projects\iree-build\install\bin"
TRANSLATE_TOOL="${INSTALL_ROOT?}/iree-translate.exe"
EMBED_DATA_TOOL="${INSTALL_ROOT?}/generate_embed_data.exe"
INPUT_NAME="simple_abs"
INPUT_PATH="${ROOT_DIR?}/iree/samples/models/simple_abs.mlir"

echo "=== Translating MLIR to Wasm VM flatbuffer output (.vmfb) ==="
${TRANSLATE_TOOL?} ${INPUT_PATH} \
  --iree-mlir-to-vm-bytecode-module \
  --iree-input-type=mhlo \
  --iree-hal-target-backends=llvm \
  --iree-llvm-target-triple=wasm32-unknown-emscripten \
  --iree-llvm-target-cpu-features=+atomics,+bulk-memory,+simd128 \
  --iree-llvm-link-embedded=false \
  --o ${BINARY_DIR}/${INPUT_NAME}.vmfb

###############################################################################
# Build the web artifacts using Emscripten                                    #
###############################################################################

echo "=== Building web artifacts using Emscripten ==="

pushd ${BUILD_DIR}

# Configure using Emscripten's CMake wrapper, then build.
# Note: The sample creates a device directly, so no drivers are required.
emcmake "${CMAKE_BIN?}" -G Ninja .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_HOST_BINARY_ROOT=$PWD/../build-host/install \
  -DIREE_BUILD_EXPERIMENTAL_WEB_SAMPLES=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF

"${CMAKE_BIN?}" --build . --target \
  iree_experimental_web_sample_dynamic_sync

popd

###############################################################################
# Serve the sample using a local webserver                                    #
###############################################################################

echo "=== Copying static files (.html, .js) to the build directory ==="

cp ${SOURCE_DIR?}/index.html ${BINARY_DIR}
cp ${SOURCE_DIR?}/iree_api.js ${BINARY_DIR}
cp ${SOURCE_DIR?}/iree_worker.js ${BINARY_DIR}

echo "=== Running local webserver, open at http://localhost:8000/ ==="

python3 ${ROOT_DIR?}/scripts/local_web_server.py --directory ${BINARY_DIR}
