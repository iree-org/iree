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
#   * Host tools must be available at the $1 arg
#
# Sample usage:
#   python -m venv .venv
#   source .venv/bin/activate
#   python -m pip install iree-compiler iree-runtime
#   build_sample.sh .venv/bin && serve_sample.sh
#
# The build directory for the emscripten build is taken from the environment
# variable IREE_EMPSCRIPTEN_BUILD_DIR, defaulting to "build-emscripten".
# Designed for CI, but can be run manually. It reuses the build directory if it
# already exists.

set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)

HOST_TOOLS_BINARY_DIR="$1"
BUILD_DIR="${IREE_EMPSCRIPTEN_BUILD_DIR:-build-emscripten}"
SOURCE_DIR="${ROOT_DIR}/experimental/web/sample_static"
BINARY_DIR="${BUILD_DIR}/experimental/web/sample_static/"
IREE_PYTHON3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE:-$(which python3)}"

###############################################################################
# Setup and checking for dependencies                                         #
###############################################################################

if ! command -v emcmake &> /dev/null
then
  echo "'emcmake' not found, setup environment according to https://emscripten.org/docs/getting_started/downloads.html"
  exit 1
fi

source "${ROOT_DIR}/build_tools/cmake/setup_build.sh"

mkdir -p "${BINARY_DIR}"

###############################################################################
# Compile from .mlir input to static C source files using host tools          #
###############################################################################

INPUT_NAME="mnist"
INPUT_PATH="${ROOT_DIR}/samples/models/mnist.mlir"

echo "=== Compiling MLIR to static library output (.vmfb, .h, .o) ==="
"${HOST_TOOLS_BINARY_DIR}/iree-compile" "${INPUT_PATH}" \
  --iree-input-type=stablehlo \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=wasm32-unknown-unknown \
  --iree-llvmcpu-target-cpu-features=+simd128 \
  --iree-llvmcpu-link-static \
  --iree-llvmcpu-static-library-output-path="${BINARY_DIR}/${INPUT_NAME}_static.o" \
  --o "${BINARY_DIR}/${INPUT_NAME}.vmfb"

echo "=== Embedding bytecode module (.vmfb) into C source files (.h, .c) ==="
"${HOST_TOOLS_BINARY_DIR}/iree-c-embed-data" "${BINARY_DIR}/${INPUT_NAME}.vmfb" \
  --output_header="${BINARY_DIR}/${INPUT_NAME}_bytecode.h" \
  --output_impl="${BINARY_DIR}/${INPUT_NAME}_bytecode.c" \
  --identifier="iree_static_${INPUT_NAME}" \
  --flatten

###############################################################################
# Build the web artifacts using Emscripten                                    #
###############################################################################

echo "=== Building web artifacts using Emscripten ==="

# Configure using Emscripten's CMake wrapper, then build.
# Note: The sample creates a device directly, so no drivers are required.
emcmake "${CMAKE_BIN}" \
  -G Ninja \
  -B "${BUILD_DIR}" \
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_HOST_BIN_DIR="${HOST_TOOLS_BINARY_DIR}" \
  -DIREE_BUILD_EXPERIMENTAL_WEB_SAMPLES=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF \
  .

"${CMAKE_BIN}" --build "${BUILD_DIR}" --target \
    iree_experimental_web_sample_static_sync \
    iree_experimental_web_sample_static_multithreaded

echo "=== Copying static files to the build directory ==="

cp "${SOURCE_DIR}/index.html" "${BINARY_DIR}"
cp "${ROOT_DIR}/docs/website/docs/assets/images/IREE_Logo_Icon_Color.svg" "${BINARY_DIR}"
cp "${SOURCE_DIR}/iree_api.js" "${BINARY_DIR}"
cp "${SOURCE_DIR}/iree_worker.js" "${BINARY_DIR}"

EASELJS_LIBRARY="${BINARY_DIR}/easeljs.min.js"
test -f "${EASELJS_LIBRARY}" || \
    wget https://code.createjs.com/1.0.0/easeljs.min.js -O "${EASELJS_LIBRARY}"
