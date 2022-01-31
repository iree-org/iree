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

BUILD_DIR=${ROOT_DIR?}/build-emscripten
mkdir -p ${BUILD_DIR}

BINARY_DIR=${BUILD_DIR}/experimental/sample_web_static/
mkdir -p ${BINARY_DIR}

###############################################################################
# Compile from .mlir input to static C source files using host tools          #
###############################################################################

# TODO(scotttodd): portable path ... discover from python install if on $PATH?
INSTALL_ROOT="D:\dev\projects\iree-build\install\bin"
TRANSLATE_TOOL="${INSTALL_ROOT?}/iree-translate.exe"
EMBED_DATA_TOOL="${INSTALL_ROOT?}/generate_embed_data.exe"
INPUT_NAME="mnist"
INPUT_PATH="${ROOT_DIR?}/iree/samples/models/mnist.mlir"

echo "=== Translating MLIR to static library output (.vmfb, .h, .o) ==="
${TRANSLATE_TOOL?} ${INPUT_PATH} \
  --iree-mlir-to-vm-bytecode-module \
  --iree-input-type=mhlo \
  --iree-hal-target-backends=llvm \
  --iree-llvm-target-triple=wasm32-unknown-unknown \
  --iree-llvm-link-embedded=false \
  --iree-llvm-link-static \
  --iree-llvm-static-library-output-path=${BINARY_DIR}/${INPUT_NAME}_static.o \
  --o ${BINARY_DIR}/${INPUT_NAME}.vmfb

echo "=== Embedding bytecode module (.vmfb) into C source files (.h, .c) ==="
${EMBED_DATA_TOOL?} ${BINARY_DIR}/${INPUT_NAME}.vmfb \
  --output_header=${BINARY_DIR}/${INPUT_NAME}_bytecode.h \
  --output_impl=${BINARY_DIR}/${INPUT_NAME}_bytecode.c \
  --identifier=iree_static_${INPUT_NAME} \
  --flatten

###############################################################################
# Build the web artifacts using Emscripten                                    #
###############################################################################

echo "=== Building web artifacts using Emscripten ==="

pushd ${ROOT_DIR?}/build-emscripten

# Configure using Emscripten's CMake wrapper, then build.
# Note: The sample creates a task device directly, so no drivers are required,
#       but some targets are gated on specific CMake options.
emcmake "${CMAKE_BIN?}" -G Ninja .. \
  -DIREE_HOST_BINARY_ROOT=$PWD/../build-host/install \
  -DIREE_BUILD_EXPERIMENTAL_WEB_SAMPLES=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_DYLIB=ON \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF

"${CMAKE_BIN?}" --build . --target \
  iree_experimental_sample_web_static_sync
  # iree_experimental_sample_web_static_multithreaded
popd

###############################################################################
# Serve the demo using a local webserver                                      #
###############################################################################

echo "=== Copying static files (index.html) to the build directory ==="

cp ${ROOT_DIR?}/experimental/sample_web_static/index.html ${BINARY_DIR}

echo "=== Running local webserver ==="
echo "    open at http://localhost:8000/build-emscripten/experimental/sample_web_static/"

# **Note**: this serves from the root so source maps can reference code in the
# source tree. A real deployment would bundle the output artifacts and serve
# them from a build/release directory.

# local_server.py is needed when using SharedArrayBuffer, with multithreading
# python3 local_server.py --directory ${ROOT_DIR?}

# http.server on its own is fine for single threaded use, and this doesn't
# break CORS for external resources like easeljs from a CDN
python3 -m http.server --directory ${ROOT_DIR?}
