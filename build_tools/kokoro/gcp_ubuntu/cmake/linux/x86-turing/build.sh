#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build the project with cmake using Kokoro.

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

# Print NVIDIA GPU information inside the docker
dpkg -l | grep nvidia
nvidia-smi || true

./build_tools/kokoro/gcp_ubuntu/check_vulkan.sh

# TODO(gcmn): It would be nice to be able to build and test as much as possible,
# so a build failure only prevents building/testing things that depend on it and
# we can still run the other tests.
echo "Building with cmake"

ROOT_DIR=$(git rev-parse --show-toplevel)

cd ${ROOT_DIR?}
rm -rf build/

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}

"$CMAKE_BIN" --version
ninja --version

mkdir build
cd build

CMAKE_ARGS=(
  "-G" "Ninja"
  # Let's make linking fast
  "-DIREE_ENABLE_LLD=ON"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"

  "-DIREE_BUILD_PYTHON_BINDINGS=ON"

  "-DIREE_ENABLE_ASSERTIONS=ON"

  # Enable CUDA backend to test on Turing hardware.
  "-DIREE_TARGET_BACKEND_CUDA=ON"
  "-DIREE_HAL_DRIVER_CUDA=ON"
)

"$CMAKE_BIN" "${CMAKE_ARGS[@]?}" "$@" ..
"$CMAKE_BIN" --build . -- -k 0

cd ${ROOT_DIR?}

export IREE_VULKAN_F16_DISABLE=0
export IREE_CUDA_DISABLE=0
echo "Testing with ctest"
./build_tools/cmake/test.sh
