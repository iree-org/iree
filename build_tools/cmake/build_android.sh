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

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}

"${CMAKE_BIN}" --version
"${CC}" --version
"${CXX}" --version
ninja --version
echo "Android NDK path: ${ANDROID_NDK}"

ROOT_DIR=$(git rev-parse --show-toplevel)
cd ${ROOT_DIR}

# BUILD the iree-import-tflite binary for importing models to benchmark from
# TFLite flatbuffers.
cd "${ROOT_DIR}/integrations/tensorflow"
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
BAZEL_BINDIR="$(${BAZEL_CMD[@]} info bazel-bin)"
"${BAZEL_CMD[@]}" build //iree_tf_compiler:iree-import-tflite \
      --config=generic_clang \
      --config=remote_cache_tf_integrations

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
  -DIREE_BUILD_TFLITE_COMPILER=ON \
  -DIREE_BUILD_SAMPLES=OFF
"${CMAKE_BIN}" --build . --target install
# Also make sure that we can generate artifacts for benchmarking on Android.
"${CMAKE_BIN}" --build . --target iree-benchmark-suites
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Build for the target (Android).

cd ${ROOT_DIR}

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
"${CMAKE_BIN}" --build .
