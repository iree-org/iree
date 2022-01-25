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
export BUILD_RISCV_DIR="${ROOT_DIR?}/build-riscv-rv32-baremetal"
export RISCV_CONFIG="rv32-baremetal"

echo "Cross-compiling with cmake"
./build_tools/cmake/build_riscv.sh

echo "Run sanity tests"
./build_tools/kokoro/gcp_ubuntu/cmake/baremetal/riscv32/test.sh
