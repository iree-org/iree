#!/bin/bash
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the IREE project towards Android with CMake. Designed for CI,
# but can be run manually. This uses previously cached build results and does
# not clear build directories.
#
# Host binaries (e.g. compiler tools) will be built and installed in build-host/
# Android binaries (e.g. tests) will be built in build-android/.

set -xeuo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <android-abi>"
  exit 1
fi

ANDROID_ABI=$1

ROOT_DIR=$(git rev-parse --show-toplevel)

CMAKE_BIN="${CMAKE_BIN:-$(which cmake)}"
"${CMAKE_BIN}" --version
"${CC}" --version
"${CXX}" --version
ninja --version

cd "${ROOT_DIR}"

# --------------------------------------------------------------------------- #
# Build for the host.

if [ -d "build-host" ]
then
  echo "build-host directory already exists. Will use cached results there."
else
  echo "build-host directory does not already exist. Creating a new one."
  mkdir build-host
fi
cd build-host

# Configure, build, install.
"${CMAKE_BIN}" -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DIREE_ENABLE_ASSERTIONS=ON \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_LEGACY_BENCHMARKS=OFF \
  -DIREE_BUILD_SAMPLES=OFF
"${CMAKE_BIN}" --build . --target install -- -k 0
# --------------------------------------------------------------------------- #

cd "${ROOT_DIR}"

# --------------------------------------------------------------------------- #
# Build for the target (Android).

if [ -d "build-android" ]
then
  echo "build-android directory already exists. Will use cached results there."
else
  echo "build-android directory does not already exist. Creating a new one."
  mkdir build-android
fi
cd build-android

# Configure towards 64-bit Android 10, then build.
"${CMAKE_BIN}" -G Ninja .. \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DANDROID_PLATFORM=android-29 \
  -DIREE_HOST_BIN_DIR="${PWD}/../build-host/install/bin" \
  -DIREE_ENABLE_ASSERTIONS=ON \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_BUILD_SAMPLES=OFF

echo "Building all for device"
echo "------------"
"${CMAKE_BIN}" --build . -- -k 0

echo "Building test deps for device"
echo "------------------"
"$CMAKE_BIN" --build . --target iree-test-deps -- -k 0
