#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds tests, running host tools, Emscripten, and CMake as needed.
#
# Prerequisites:
#   * Environment must be configured for Emscripten
#   * Host tools must be built (default at IREE_SOURCE_DIR/build-host/install).
#     The build_tools/cmake/build_host_install.sh script can do this for you.
#
# Usage:
#   build_tests.sh (optional install path) && serve_tests.sh

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
SOURCE_DIR=${ROOT_DIR}/experimental/web/testing

BUILD_DIR=${ROOT_DIR?}/build-emscripten
mkdir -p ${BUILD_DIR}

BINARY_DIR=${BUILD_DIR}/experimental/web/testing/
mkdir -p ${BINARY_DIR}

INSTALL_ROOT="${1:-${ROOT_DIR}/build-host/install}"

###############################################################################
# Build the web artifacts using Emscripten                                    #
###############################################################################

echo "=== Building web artifacts using Emscripten ==="

pushd ${ROOT_DIR?}/build-emscripten

# Configure using Emscripten's CMake wrapper, then build.
emcmake "${CMAKE_BIN?}" -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DIREE_HOST_BINARY_ROOT=${INSTALL_ROOT} \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_ENABLE_CPUINFO=OFF \
  -DIREE_BUILD_TESTS=ON

"${CMAKE_BIN?}" --build .

popd

echo "=== Copying static files to the build directory ==="

cp ${SOURCE_DIR?}/index.html ${BINARY_DIR}
cp ${ROOT_DIR?}/docs/website/overrides/ghost.svg ${BINARY_DIR}
