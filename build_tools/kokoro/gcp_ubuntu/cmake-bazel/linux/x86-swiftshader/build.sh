#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and test the python bindings and frontend integrations.

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

./build_tools/kokoro/gcp_ubuntu/check_vulkan.sh

# Print SwiftShader git commit
cat /swiftshader/git-commit

echo "Initializing submodules"
./scripts/git/submodule_versions.py init

# BUILD the integrations binaries with Bazel and run any lit tests
pushd integrations/tensorflow
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
BAZEL_BINDIR="$(${BAZEL_CMD[@]?} info bazel-bin)"
"${BAZEL_CMD[@]?}" query //iree_tf_compiler/... | \
   xargs "${BAZEL_CMD[@]?}" test --config=generic_clang \
      --test_tag_filters="-nokokoro" \
      --build_tag_filters="-nokokoro"
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

export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

echo "Testing with CTest"
ctest --timeout 900 --output-on-failure \
   --tests-regex "^integrations/tensorflow/|^bindings/python/" \
   --label-exclude "^nokokoro$|^vulkan_uses_vk_khr_shader_float16_int8$"
