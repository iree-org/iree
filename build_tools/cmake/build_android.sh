# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Cross-compile the IREE project towards Android with CMake. Designed for CI,
# but can be run manually. This uses previously cached build results and does
# not clear build directories.
#
# Host binaries (e.g. compiler tools) will be built and installed in build-host/
# Android binaries (e.g. tests) will be built in build-android/.

set -x
set -e

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <android-abi>"
  exit 1
fi

ANDROID_ABI=$1

ROOT_DIR=$(git rev-parse --show-toplevel)

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}

"${CMAKE_BIN?}" --version
ninja --version

cd ${ROOT_DIR?}

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
"${CMAKE_BIN?}" -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF
"${CMAKE_BIN?}" --build . --target install
# --------------------------------------------------------------------------- #

cd ${ROOT_DIR?}

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
"${CMAKE_BIN?}" -G Ninja .. \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK?}/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="${ANDROID_ABI?}" \
  -DANDROID_PLATFORM=android-29 \
  -DIREE_HOST_BINARY_ROOT=$PWD/../build-host/install \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_ENABLE_MLIR=OFF \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_BUILD_SAMPLES=OFF
"${CMAKE_BIN?}" --build .
