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
#     The build_tools/cmake/build_host_tools.sh script can do this for you.
#
# Usage:
#   build_sample.sh (optional install path) && serve_sample.sh

set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)

HOST_BUILD_DIR="${IREE_HOST_BUILD_DIR:-${ROOT_DIR}/build-host}"
BUILD_DIR="${IREE_EMPSCRIPTEN_BUILD_DIR:-build-emscripten}"
INSTALL_ROOT="$(realpath ${1:-${HOST_BUILD_DIR}/install})"
SOURCE_DIR="${ROOT_DIR}/experimental/web/sample_webgpu"
BINARY_DIR="${BUILD_DIR}/experimental/web/sample_webgpu"

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

echo "=== Compiling sample MLIR files to VM FlatBuffer outputs (.vmfb) ==="
COMPILE_TOOL="${INSTALL_ROOT?}/bin/iree-compile"

# TODO(#11321): Enable iree-codegen-gpu-native-math-precision by default?
compile_sample() {
  echo "  Compiling '$1' sample for WebGPU..."
  ${COMPILE_TOOL?} $3 \
    --iree-input-type=$2 \
    --iree-hal-target-backends=webgpu-spirv \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-stream-resource-alias-mutable-bindings=true \
    --o ${BINARY_DIR}/$1_webgpu.vmfb
}

compile_sample "simple_abs"       "none"      "${ROOT_DIR?}/samples/models/simple_abs.mlir"
compile_sample "multiple_results" "none"      "${SOURCE_DIR?}/multiple_results.mlir"
compile_sample "fullyconnected"   "stablehlo" "${ROOT_DIR?}/tests/e2e/stablehlo_models/fullyconnected.mlir"

# Does not run yet (uses internal readback, which needs async buffer mapping?)
# compile_sample "collatz" "stablehlo" "${ROOT_DIR?}/tests/e2e/stablehlo_models/collatz.mlir"

# Slow, so just run on demand
# TODO(scotttodd): iree-import-tflite (see generate_web_metrics.sh script)
# compile_sample "mobilebert" "tosa" "MobileBertSquad_fp32.mlir"
# compile_sample "posenet"    "tosa" "Posenet_fp32.mlir"
# compile_sample "mobilessd"  "tosa" "MobileSSD_fp32.mlir"

###############################################################################
# Build the web artifacts using Emscripten                                    #
###############################################################################

echo "=== Building web artifacts using Emscripten ==="

pushd ${BUILD_DIR}

# Configure using Emscripten's CMake wrapper, then build.
# Note: The sample creates a device directly, so no drivers are required.
emcmake "${CMAKE_BIN?}" -G Ninja .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_HOST_BIN_DIR="${INSTALL_ROOT}/bin" \
  -DIREE_BUILD_EXPERIMENTAL_WEB_SAMPLES=ON \
  -DIREE_ENABLE_THREADING=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=OFF \
  -DIREE_HAL_DRIVER_LOCAL_TASK=OFF \
  -DIREE_EXTERNAL_HAL_DRIVERS=webgpu \
  -DIREE_ENABLE_ASAN=OFF \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF

"${CMAKE_BIN?}" --build . --target \
  iree_experimental_web_sample_webgpu

popd

echo "=== Copying static files (.html, .js) to the build directory ==="

cp ${SOURCE_DIR?}/index.html ${BINARY_DIR}
cp "${ROOT_DIR}/docs/website/docs/assets/images/ghost.svg" "${BINARY_DIR}"
cp ${SOURCE_DIR?}/iree_api_webgpu.js ${BINARY_DIR}
