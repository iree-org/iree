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

echo "Initializing submodules"
git submodule update --init --jobs 8 --depth 1

./build_tools/kokoro/gcp_ubuntu/check_vulkan.sh

# Print SwiftShader git commit
cat /swiftshader/git-commit

# BUILD the integrations binaries with Bazel and run any lit tests
IREE_SRC_DIR="$PWD"
pushd integrations/tensorflow
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
BAZEL_BINDIR="$(${BAZEL_CMD[@]?} info bazel-bin)"
# xargs is set to high arg limits to avoid multiple Bazel invocations and will
# hard fail if the limits are exceeded.
# See https://github.com/bazelbuild/bazel/issues/12479
"${BAZEL_CMD[@]?}" query //iree_tf_compiler/... | \
   xargs --max-args 1000000 --max-chars 1000000 --exit \
    "${BAZEL_CMD[@]?}" test \
      --config=remote_cache_bazel_ci \
      --config=generic_clang \
      --test_tag_filters="-nokokoro" \
      --build_tag_filters="-nokokoro"
bash ./symlink_binaries.sh
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
   -DIREE_BUILD_PYTHON_BINDINGS=ON \
   -DIREE_HAL_DRIVER_CUDA=ON \
   -DIREE_TARGET_BACKEND_CUDA=ON \
   .

echo "Building with Ninja"
cd "${CMAKE_BUILD_DIR?}"
ninja

export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

tests_passed=true

echo "***** Testing with CTest *****"
if ! ctest --timeout 900 --output-on-failure \
   --tests-regex "^integrations/tensorflow/|^runtime/bindings/python/" \
   --label-exclude "^nokokoro$|^vulkan_uses_vk_khr_shader_float16_int8$"
then
   tests_passed=false
fi

echo "***** Running TensorFlow integration tests *****"
# TODO: Use "--timeout 900" instead of --max-time below. Requires that
# `psutil` python package be installed in the VM for per test timeout.
cd "$IREE_SRC_DIR"
source "${CMAKE_BUILD_DIR?}/.env" && export PYTHONPATH
LIT_SCRIPT="$IREE_SRC_DIR/third_party/llvm-project/llvm/utils/lit/lit.py"
if ! python3 "$LIT_SCRIPT" -v integrations/tensorflow/test \
   --max-time 1800 \
   -D FEATURES=vulkan
then
   tests_passed=false
fi

if ! $tests_passed; then
   echo "Some tests failed!!!"
   exit 1
fi
