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

# Build and test the python bindings and frontend integrations on GPU.

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
python3 -c 'import jax; print(jax.__version__)'

# Print NVIDIA GPU information inside the docker
dpkg -l | grep nvidia
nvidia-smi || true

./build_tools/kokoro/gcp_ubuntu/check_vulkan.sh

echo "Initializing submodules"
./scripts/git/submodule_versions.py init

# BUILD the integrations binaries with Bazel
pushd integrations/tensorflow
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
BAZEL_BINDIR="$(${BAZEL_CMD[@]?} info bazel-bin)"
"${BAZEL_CMD[@]?}" build --config=generic_clang //iree_tf_compiler:all
popd



CMAKE_BUILD_DIR="$HOME/iree/build/tf"

# TODO(gcmn): It would be nice to be able to build and test as much as possible,
# so a build failure only prevents building/testing things that depend on it and
# we can still run the other tests.
echo "Configuring CMake"
"${CMAKE_BIN}" -B "${CMAKE_BUILD_DIR?}" -G Ninja \
   -DIREE_TF_TOOLS_ROOT="${BAZEL_BINDIR?}/iree_tf_compiler/" \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DIREE_BUILD_COMPILER=ON \
   -DIREE_BUILD_TESTS=ON \
   -DIREE_BUILD_SAMPLES=OFF \
   -DIREE_BUILD_XLA_COMPILER=ON \
   -DIREE_BUILD_TFLITE_COMPILER=ON \
   -DIREE_BUILD_TENSORFLOW_COMPILER=ON .

echo "Building with Ninja"
cd "${CMAKE_BUILD_DIR?}"
ninja

# Limit parallelism dramatically to avoid exhausting GPU memory
# TODO(#5162): Handle this more robustly
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-1}

# Only test drivers that use the GPU, since we run all tests on non-GPU machines
# as well.
echo "Testing with CTest"
ctest --output-on-failure \
   --tests-regex "^integrations/tensorflow/|^bindings/python/" \
   --label-regex "^driver=vulkan$|^driver=cuda$" \
   --label-exclude "^nokokoro$"
