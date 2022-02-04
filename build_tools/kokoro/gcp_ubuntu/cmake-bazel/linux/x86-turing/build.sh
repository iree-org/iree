#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

echo "Initializing submodules"
git submodule update --init --jobs 8 --depth 1

# Print NVIDIA GPU information inside the docker
dpkg -l | grep nvidia
nvidia-smi || true

./build_tools/kokoro/gcp_ubuntu/check_vulkan.sh

# BUILD the integrations binaries with Bazel
IREE_SRC_DIR="$PWD"
pushd integrations/tensorflow
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
BAZEL_BINDIR="$(${BAZEL_CMD[@]?} info bazel-bin)"
# Technically, the cache key of the `remote_cache_bazel_ci` config
# references the Docker image used by the non-GPU build that includes
# swiftshader. In practice though, the part that matters for building these
# binaries is common to both of them.
# TODO(gcmn): Split this out into a multistage build so we only build these
# once anyway.
"${BAZEL_CMD[@]?}" build \
   --config=generic_clang \
   --config=remote_cache_bazel_ci \
   //iree_tf_compiler:all
bash ./symlink_binaries.sh
popd



CMAKE_BUILD_DIR="$HOME/iree/build/tf"

# TODO(gcmn): It would be nice to be able to build and test as much as possible,
# so a build failure only prevents building/testing things that depend on it and
# we can still run the other tests.
# TODO: Add "-DIREE_TARGET_BACKEND_CUDA=ON -DIREE_HAL_DRIVER_CUDA=ON" once the
# VMs have been updated with the correct CUDA SDK.
echo "Configuring CMake"
"${CMAKE_BIN}" -B "${CMAKE_BUILD_DIR?}" -G Ninja \
   -DIREE_TF_TOOLS_ROOT="${BAZEL_BINDIR?}/iree_tf_compiler/" \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DIREE_BUILD_COMPILER=ON \
   -DIREE_BUILD_TESTS=ON \
   -DIREE_BUILD_SAMPLES=OFF \
   -DIREE_BUILD_PYTHON_BINDINGS=ON \
   .

echo "Building with Ninja"
cd "${CMAKE_BUILD_DIR?}"
ninja

# Limit parallelism dramatically to avoid exhausting GPU memory
# TODO(#5162): Handle this more robustly
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-1}

tests_passed=true

# Only test drivers that use the GPU, since we run all tests on non-GPU machines
# as well.
echo "***** Testing with CTest *****"
if ! ctest --timeout 900 --output-on-failure \
   --tests-regex "^integrations/tensorflow/|^bindings/python/" \
   --label-regex "^driver=vulkan$|^driver=cuda$" \
   --label-exclude "^nokokoro$"
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
   -D DISABLE_FEATURES=llvmaot \
   -D FEATURES=vulkan
then
   tests_passed=false
fi

if ! $tests_passed; then
   echo "Some tests failed!!!"
   exit 1
fi
