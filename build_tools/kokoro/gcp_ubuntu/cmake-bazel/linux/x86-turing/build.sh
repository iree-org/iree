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

# Build and test the python bindings and tensorflow integrations on GPU.

set -e
set -x

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Check these exist and print the versions for later debugging
export CMAKE_BIN="$(which cmake)"
"${CMAKE_BIN?}" --version
"${CC?}" --version
"${CXX?}" --version
python3 --version
python3 -c 'import tensorflow as tf; print(tf.__version__)'

./build_tools/kokoro/gcp_ubuntu/check_vulkan.sh

# Print SwiftShader git commit
cat /swiftshader/git-commit

echo "Initializing submodules"
./scripts/git/submodule_versions.py init

# TODO(gcmn): It would be nice to be able to build and test as much as possible,
# so a build failure only prevents building/testing things that depend on it and
# we can still run the other tests.
echo "Configuring CMake"
"${CMAKE_BIN}" -B ../iree-build/ -G Ninja \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DIREE_BUILD_COMPILER=ON \
   -DIREE_BUILD_TESTS=ON \
   -DIREE_BUILD_SAMPLES=OFF \
   -DIREE_BUILD_DEBUGGER=OFF \
   -DIREE_BUILD_TENSORFLOW_COMPILER=ON .

echo "Building with Ninja"
cd ../iree-build/
ninja

echo "Testing with CTest"
ctest -R 'tensorflow_e2e|bindings/python'
