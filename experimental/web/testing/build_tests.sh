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
#     The build_tools/cmake/build_host_tools.sh script can do this for you.
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

BINARY_DIR=${BUILD_DIR}/experimental/web/testing
mkdir -p ${BINARY_DIR}

INSTALL_ROOT="${1:-${ROOT_DIR}/build-host/install}"

###############################################################################
# Build the web artifacts using Emscripten                                    #
###############################################################################

echo "=== Building web artifacts using Emscripten ==="

pushd ${ROOT_DIR?}/build-emscripten > /dev/null

# Configure using Emscripten's CMake wrapper, then build.
emcmake "${CMAKE_BIN?}" -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_HOST_BIN_DIR="${INSTALL_ROOT}/bin" \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF \
    -DIREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE=ON \
    -DIREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS=OFF \
    -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_ENABLE_CPUINFO=OFF \
    -DIREE_ENABLE_ASAN=OFF \
    -DIREE_BUILD_TESTS=ON

echo "=== Building default targets ==="
"${CMAKE_BIN}" --build . -- -k 0

echo "=== Building test deps ==="
"${CMAKE_BIN?}" --build . --target iree-test-deps -- -k 0

echo "=== Building sample deps ==="
"${CMAKE_BIN?}" --build . --target iree-sample-deps -- -k 0

echo "=== Generating list of tests ==="

# TODO(scotttodd): Move this parsing and substitution into CMake?
ctest --show-only=json-v1 > ctest_dump.json
python3 ${SOURCE_DIR?}/parse_test_list.py \
    --ctest_dump=ctest_dump.json \
    --build_dir=. \
    --output_format=html \
    -o test_list.html
# Substitute {{TEST_LIST}} in the template for the contents of test_list.html
#   https://unix.stackexchange.com/a/49438
#   https://unix.stackexchange.com/a/141398
sed -e '/{{TEST_LIST}}/ {' \
    -e 'r test_list.html' \
    -e 'd' \
    -e '}' \
    ${SOURCE_DIR?}/index_template.html \
    > ${BINARY_DIR}/index.html

popd > /dev/null

echo "=== Copying static files to the build directory ==="

cp ${SOURCE_DIR?}/test-runner.html ${BINARY_DIR}
cp ${SOURCE_DIR?}/*.js ${BINARY_DIR}
cp ${ROOT_DIR?}/docs/website/overrides/.icons/iree/ghost.svg ${BINARY_DIR}
