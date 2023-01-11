#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Builds the benchmark tools for multiple linux targets. The first argument
# specifies which targets should be built (E.g., linux-x86_64;linux-riscv). Put
# "all" or empty to build all targets.

set -xeuo pipefail

# Print the UTC time when set -x is on.
export PS4='[$(date -u "+%T %Z")] '

TARGETS="${1:-all}"

# Check these exist and print the versions for later debugging.
CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
"${CMAKE_BIN}" --version
"${CC}" --version
"${CXX}" --version
ninja --version
python3 --version

echo "Initializing submodules"
git submodule sync
git submodule update --init --jobs 8 --depth 1

ROOT_DIR=$(git rev-parse --show-toplevel)
cd "${ROOT_DIR}"

# --------------------------------------------------------------------------- #
# Build for the target (linux-x86_64).
if [[ "${TARGETS}" =~ linux-x86_64|all ]]; then
  cd "${ROOT_DIR}"

  if [ -d "build-targets/linux-x86_64" ]
  then
    echo "linux-x86_64 directory already exists. Will use cached results there."
  else
    echo "linux-x86_64 directory does not already exist. Creating a new one."
    mkdir -p build-targets/linux-x86_64
  fi
  cd build-targets/linux-x86_64

  "${CMAKE_BIN}" -G Ninja ../.. \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_BUILD_TESTS=OFF \
    -DIREE_BUILD_SAMPLES=OFF

  "${CMAKE_BIN}" --build . --target iree-benchmark-module -- -k 0
fi
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Build for the target (linux-cuda).
if [[ "${TARGETS}" =~ linux-cuda|all ]]; then
  cd "${ROOT_DIR}"

  if [ -d "build-targets/linux-cuda" ]
  then
    echo "linux-cuda directory already exists. Will use cached results there."
  else
    echo "linux-cuda directory does not already exist. Creating a new one."
    mkdir -p build-targets/linux-cuda
  fi
  cd build-targets/linux-cuda

  "${CMAKE_BIN}" -G Ninja ../.. \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_BUILD_TESTS=OFF \
    -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_HAL_DRIVER_CUDA=ON

  "${CMAKE_BIN}" --build . --target iree-benchmark-module -- -k 0
fi
# --------------------------------------------------------------------------- #
