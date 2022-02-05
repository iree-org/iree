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
# TODO: Add "-DIREE_TARGET_BACKEND_CUDA=ON -DIREE_HAL_DRIVER_CUDA=ON" once the
# VMs have been updated with the correct CUDA SDK.
echo "Building with cmake"

ROOT_DIR=$(git rev-parse --show-toplevel)

cd ${ROOT_DIR?}
rm -rf build/

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

CMAKE_ARGS=(
  "-G" "Ninja"
  # Let's make linking fast
  "-DIREE_ENABLE_LLD=ON"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"

  # Enable docs build on the CI. The additional builds are pretty fast and
  # give us early warnings for some types of website publication errors.
  "-DIREE_BUILD_DOCS=ON"

  # Enable building the python bindings on CI. Most heavy targets are gated on
  # IREE_ENABLE_TENSORFLOW, so what's left here should be fast.
  "-DIREE_BUILD_PYTHON_BINDINGS=ON"

  # Enable assertions. We wouldn't want to be testing *only* with assertions
  # enabled, but at the moment only certain CI builds are using this script,
  # e.g. ASan builds are not using this, so by enabling assertions here, we
  # get a reasonable mix of {builds with asserts, builds with other features
  # such as ASan but without asserts}.
  "-DIREE_ENABLE_ASSERTIONS=ON"

  "-DIREE_TARGET_BACKEND_CUDA=ON"

  "-DIREE_HAL_DRIVER_CUDA=ON"
)

"$CMAKE_BIN" "${CMAKE_ARGS[@]?}" "$@" ..
"$CMAKE_BIN" --build .


export IREE_VULKAN_F16_DISABLE=0
export IREE_CUDA_DISABLE=0
echo "Testing with ctest"
./build_tools/cmake/test.sh
