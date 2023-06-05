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
SOURCE_DIR=${ROOT_DIR}/experimental/web/sample_webgpu

BUILD_DIR=${ROOT_DIR?}/build-emscripten
mkdir -p ${BUILD_DIR}

BINARY_DIR=${BUILD_DIR}/experimental/web/sample_webgpu
mkdir -p ${BINARY_DIR}

INSTALL_ROOT="${1:-${ROOT_DIR}/build-host/install}"

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
    --iree-hal-target-backends=webgpu \
    --iree-codegen-gpu-native-math-precision=true \
    --o ${BINARY_DIR}/$1_webgpu.vmfb
}

compile_sample "simple_abs"     "none" "${ROOT_DIR?}/samples/models/simple_abs.mlir"
compile_sample "fullyconnected" "mhlo" "${ROOT_DIR?}/tests/e2e/models/fullyconnected.mlir"

# Does not run yet (uses internal readback, which needs async buffer mapping?)
# compile_sample "collatz"        "${ROOT_DIR?}/tests/e2e/models/collatz.mlir"

# Slow, so just run on demand
# compile_sample "mobilebert" "tosa" "D:/dev/projects/iree-data/models/2022_10_28/mobilebertsquad.tflite.mlir"
# compile_sample "posenet"    "tosa" "D:/dev/projects/iree-data/models/2022_10_28/posenet.tflite.mlir"
# compile_sample "mobilessd"  "tosa" "D:/dev/projects/iree-data/models/2022_10_28/mobile_ssd_v2_float_coco.tflite.mlir"

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
cp "${ROOT_DIR}/docs/website/overrides/.icons/iree/ghost.svg" "${BINARY_DIR}"
cp ${SOURCE_DIR?}/iree_api_webgpu.js ${BINARY_DIR}
