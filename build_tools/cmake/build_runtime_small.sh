#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build IREE's runtime using CMake. Designed for CI, but can be run manually.
# This uses previously cached build results and does not clear build
# directories.

set -e
set -x

ROOT_DIR=$(git rev-parse --show-toplevel)
cd ${ROOT_DIR?}

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
"${CMAKE_BIN?}" --version
ninja --version

if [ -d "build-runtime-small" ]
then
  echo "build-runtime-small directory already exists. Will use cached results there."
else
  echo "build-runtime-small directory does not already exist. Creating a new one."
  mkdir build-runtime-small
fi
cd build-runtime-small

"${CMAKE_BIN?}" -G Ninja .. \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DIREE_SIZE_OPTIMIZED=ON \
  -DIREE_BUILD_COMPILER=OFF
"${CMAKE_BIN?}" --build . -- -k 0
