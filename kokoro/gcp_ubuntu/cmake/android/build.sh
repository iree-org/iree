#!/bin/bash

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

# Cross-compile the project towards Android with CMake using Kokoro.

set -e
set -x

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <android-abi>"
  exit 1
fi

ANDROID_ABI=$1

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Check these exist and print the versions for later debugging
export CMAKE_BIN="$(which cmake)"
"$CMAKE_BIN" --version
"$CC" --version
"$CXX" --version
python3 --version
echo "Android NDK path: $ANDROID_NDK"

echo "Initializing submodules"
./scripts/git/submodule_versions.py init

echo "Cross-compiling with cmake"
./build_tools/cmake/build_android.sh $ANDROID_ABI
