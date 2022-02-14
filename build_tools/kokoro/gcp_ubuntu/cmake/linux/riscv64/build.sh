#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the project towards RISCV with CMake using Kokoro.

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

echo "Initializing submodules"
git submodule update --init --jobs 8 --depth 1

export ROOT_DIR="$(git rev-parse --show-toplevel)"
export BUILD_HOST_DIR="${ROOT_DIR?}/build-host"
export BUILD_RISCV_DIR="${ROOT_DIR?}/build-riscv"

# BUILD the integrations iree-import-tflite with Bazel
echo "Build iree-import-tflite with Bazel"
pushd "${ROOT_DIR}/integrations/tensorflow"
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
BAZEL_BINDIR="$(${BAZEL_CMD[@]?} info bazel-bin)"
# xargs is set to high arg limits to avoid multiple Bazel invocations and will
# hard fail if the limits are exceeded.
# See https://github.com/bazelbuild/bazel/issues/12479
xargs --max-args 1000000 --max-chars 1000000 --exit \
  "${BAZEL_CMD[@]?}" build \
    --config=remote_cache_bazel_ci \
    --config=generic_clang \
    --build_tag_filters="-nokokoro" \
    //iree_tf_compiler:iree-import-tflite
popd

export PATH="${BAZEL_BINDIR?}/iree_tf_compiler:${PATH}"

echo "Cross-compiling with cmake"
./build_tools/cmake/build_riscv.sh

echo "Run sanity tests"
./build_tools/kokoro/gcp_ubuntu/cmake/linux/riscv64/test.sh
