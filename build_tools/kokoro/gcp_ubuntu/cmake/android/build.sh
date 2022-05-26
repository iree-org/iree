#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the project towards Android with CMake using Kokoro. First
# builds the TFLite import binary with Bazel to allow testing the build of
# benchmarks for Android.

set -xeuo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <android-abi>"
  exit 1
fi

ANDROID_ABI="$1"

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

echo "Initializing submodules"
git submodule update --init --jobs 8 --depth 1

ROOT_DIR="$(git rev-parse --show-toplevel)"

CMAKE_BIN="${CMAKE_BIN:-$(which cmake)}"

# Check these exist and print the versions for later debugging
"${CMAKE_BIN}" --version
bazel --version
"${CC?}" --version
"${CXX?}" --version
python3 --version
ninja --version
echo "Android NDK path: $ANDROID_NDK"

cd "${ROOT_DIR}"

# BUILD the iree-import-tflite binary for importing models to benchmark from
# TFLite FlatBuffers.
cd "${ROOT_DIR}/integrations/tensorflow"
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
BAZEL_BINDIR="$(${BAZEL_CMD[@]} info bazel-bin)"
"${BAZEL_CMD[@]}" build //iree_tf_compiler:iree-import-tflite \
      --config=generic_clang \
      --config=remote_cache_bazel_ci
# So the benchmark build below can find the importer binaries that were built.
export PATH="$PWD/bazel-bin/iree_tf_compiler:$PATH"

# --------------------------------------------------------------------------- #
# Build for the host.

cd "${ROOT_DIR}"

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
  -DIREE_BUILD_BENCHMARKS=ON \
  -DIREE_BUILD_SAMPLES=OFF

"${CMAKE_BIN}" --build . --target install -- -k 0
# Also make sure that we can generate artifacts for benchmarking on Android.
"${CMAKE_BIN}" --build . --target iree-benchmark-suites -- -k 0
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Build for the target (Android).

cd "${ROOT_DIR}"

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
  -DIREE_HOST_BINARY_ROOT="${PWD}/../build-host/install" \
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
