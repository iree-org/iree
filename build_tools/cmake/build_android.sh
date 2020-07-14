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
# not clear the build directory.

set -x
set -e

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <android-abi>"
  exit 1
fi

ANDROID_ABI=$1

ROOT_DIR=$(git rev-parse --show-toplevel)

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}

"$CMAKE_BIN" --version
ninja --version

cd ${ROOT_DIR?}
if [ -d "build" ]
then
  echo "Build directory already exists. Will use cached results there."
else
  echo "Build directory does not already exist. Creating a new one."
  mkdir build
fi

cd build

# Configure towards 64-bit Android 10.
"$CMAKE_BIN" -G Ninja .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="$ANDROID_ABI" \
  -DANDROID_PLATFORM=android-29 \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_HOST_C_COMPILER=$(which clang) \
  -DIREE_HOST_CXX_COMPILER=$(which clang++)

# TODO(#2494): Invoke once after fixing the flaky build failure in GCP.
"$CMAKE_BIN" --build . || "$CMAKE_BIN" --build .
