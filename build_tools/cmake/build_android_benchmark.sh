#!/bin/bash
# Copyright 2021 The IREE Authors
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

# Print the UTC time when set -x is on.
export PS4='[$(date -u "+%T %Z")] '

# Check these exist and print the versions for later debugging.
CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
"${CMAKE_BIN}" --version
"${CC}" --version
"${CXX}" --version
ninja --version
python3 --version
echo "Android NDK path: ${ANDROID_NDK}"

echo "Initializing submodules"
git submodule sync
git submodule update --init --jobs 8 --depth 1

ROOT_DIR=$(git rev-parse --show-toplevel)
cd "${ROOT_DIR}"

# BUILD the iree-import-tflite binary for importing models to benchmark from
# TFLite FlatBuffers.
cd "${ROOT_DIR}/integrations/tensorflow"
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
BAZEL_BINDIR="$(${BAZEL_CMD[@]} info bazel-bin)"
"${BAZEL_CMD[@]}" build \
      //iree_tf_compiler:iree-import-tflite \
      --config=generic_clang \
      --config=remote_cache_bazel_tf_ci
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
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_LEGACY_BENCHMARKS=ON \
  -DIREE_BUILD_MICROBENCHMARKS=ON \
  -DIREE_BUILD_SAMPLES=OFF

"${CMAKE_BIN}" --build . --target install -- -k 0
# Also generate artifacts for benchmarking on Android.
"${CMAKE_BIN}" --build . --target \
  iree-benchmark-suites-android-arm64-v8a \
  iree-benchmark-suites-android-adreno \
  iree-benchmark-suites-android-mali \
  -- -k 0
"${CMAKE_BIN}" --build . --target iree-microbenchmark-suites -- -k 0
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
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DIREE_HOST_BIN_DIR="${PWD}/../build-host/install/bin" \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_BUILD_SAMPLES=OFF
"${CMAKE_BIN}" --build . --target iree-benchmark-module -- -k 0

# --------------------------------------------------------------------------- #
# Build for the target (Android) with tracing.

cd "${ROOT_DIR}"

if [ -d "build-android-trace" ]
then
  echo "build-android-trace directory already exists. Will use cached results there."
else
  echo "build-android-trace directory does not already exist. Creating a new one."
  mkdir build-android-trace
fi
cd build-android-trace

# Configure towards 64-bit Android 10, then build.
"${CMAKE_BIN}" -G Ninja .. \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DIREE_HOST_BIN_DIR="${PWD}/../build-host/install/bin" \
  -DIREE_ENABLE_RUNTIME_TRACING=ON \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_BUILD_SAMPLES=OFF
"${CMAKE_BIN}" --build . --target iree-benchmark-module -- -k 0
