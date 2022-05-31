#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile IREE's TFLite Java Bindings with Gradle and CMake. Produces an
# iree-tflite-bindings-debug.aar and an iree-tflite-bindings-release.aar
# library under bindings/tflite/java/output. Designed for CI, but can be run
# manually.

set -x
set -e
set -o pipefail

# --------------------------------------------------------------------------- #
ROOT_DIR=$(git rev-parse --show-toplevel)

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}

"${CMAKE_BIN?}" --version
ninja --version
gradle --version

cd ${ROOT_DIR?}

# --------------------------------------------------------------------------- #
# Build the host libraries

HOST_BUILD_DIR=${ROOT_DIR?}/build-host
HOST_INSTALL_DIR=${HOST_BUILD_DIR}/install

cmake -G Ninja -B ${HOST_BUILD_DIR} \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DCMAKE_C_COMPILER="${CC:-clang}" \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DCMAKE_INSTALL_PREFIX=${HOST_INSTALL_DIR} .

cmake --build ${HOST_BUILD_DIR} --target install

# --------------------------------------------------------------------------- #
# Build native libraries

ANDROID_BUILD_DIR=${ROOT_DIR?}/build-android

# Todo: Support multiple ABIs. For now we build a singe abi: arm64.
ANDROID_ABI=arm64-v8a

cmake -G Ninja -B ${ANDROID_BUILD_DIR} \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DIREE_HOST_BINARY_ROOT="${HOST_INSTALL_DIR}" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DANDROID_PLATFORM="android-29" \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_BINDINGS_TFLITE=ON \
  -DIREE_BUILD_BINDINGS_TFLITE_JAVA=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=OFF  .

cmake --build  ${ANDROID_BUILD_DIR}  --target iree-tflite-bindings

NATIVE_LIBRARY=${ANDROID_BUILD_DIR}/bindings/tflite/java/org/tensorflow/lite/native/libiree-tflite-bindings.so

# --------------------------------------------------------------------------- #
# Setup the gradle build with native libraries

BINDINGS_DIR=${ROOT_DIR?}/bindings/tflite/java
cd ${BINDINGS_DIR}

# Copy the native library(s) to the jniLibs folder
mkdir -p jniLibs/${ANDROID_ABI}
cp ${NATIVE_LIBRARY} jniLibs/${ANDROID_ABI}

# --------------------------------------------------------------------------- #
# Build the Android library

gradle wrapper

# Note: since we're providing the native libraries, we omit the tasks that
# generate them from the build.gradle.
./gradlew build \
  -x externalNativeBuildDebug \
  -x externalNativeBuildCleanDebug \
  -x externalNativeBuildRelease \
  -x externalNativeBuildCleanRelease \
  -x cmakeConfigureHost \
  -x cmakeBuildHost

echo "Android Library Artifacts:"
ls build/outputs/aar/
