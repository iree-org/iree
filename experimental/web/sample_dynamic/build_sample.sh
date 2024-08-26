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
SOURCE_DIR="${ROOT_DIR}/experimental/web/sample_dynamic"
BINARY_DIR="${BUILD_DIR}/experimental/web/sample_dynamic"
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

mkdir -p ${BINARY_DIR}

###############################################################################
# Compile from .mlir input to portable .vmfb file using host tools            #
###############################################################################

compile_sample() {
  echo "  Compiling '$1' sample..."
  "${HOST_TOOLS_BINARY_DIR}/iree-compile" "$2" \
    --iree-input-type=stablehlo \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=wasm32-unknown-emscripten \
    --iree-llvmcpu-target-cpu-features=+atomics,+bulk-memory,+simd128 \
    --o "${BINARY_DIR}/$1.vmfb"
}

echo "=== Compiling sample MLIR files to VM FlatBuffer outputs (.vmfb) ==="
compile_sample "simple_abs"     "${ROOT_DIR}/samples/models/simple_abs.mlir"
compile_sample "fullyconnected" "${ROOT_DIR}/tests/e2e/stablehlo_models/fullyconnected.mlir"
compile_sample "collatz"        "${ROOT_DIR}/tests/e2e/stablehlo_models/collatz.mlir"

###############################################################################
# Build the web artifacts using Emscripten                                    #
###############################################################################

echo "=== Building web artifacts using Emscripten ==="

# Configure using Emscripten's CMake wrapper, then build.
# Note: The sample creates a device directly, so no drivers are required.
emcmake "${CMAKE_BIN}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_HOST_BIN_DIR="${HOST_TOOLS_BINARY_DIR}" \
  -DIREE_BUILD_EXPERIMENTAL_WEB_SAMPLES=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -UIREE_EXTERNAL_HAL_DRIVERS \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF \
  .

"${CMAKE_BIN}" --build "${BUILD_DIR}" --target \
  iree_experimental_web_sample_dynamic_sync

echo "=== Copying static files (.html, .js) to the build directory ==="

cp "${SOURCE_DIR}/index.html" "${BINARY_DIR}"
cp "${SOURCE_DIR}/benchmarks.html" "${BINARY_DIR}"
cp "${ROOT_DIR}/docs/website/docs/assets/images/IREE_Logo_Icon_Color.svg" "${BINARY_DIR}"
cp "${SOURCE_DIR}/iree_api.js" "${BINARY_DIR}"
cp "${SOURCE_DIR}/iree_worker.js" "${BINARY_DIR}"
