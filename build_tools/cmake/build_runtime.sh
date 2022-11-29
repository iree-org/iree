#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build only the IREE runtime using CMake
#
# Designed for CI, but can be run locally. The desired build directory can be
# passed as the first argument. Otherwise, it uses the environment variable
# IREE_RUNTIME_BUILD_DIR, defaulting to "build-runtime". It reuses the build
# directory if it already exists. Expects to be run from the root of the IREE
# repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_RUNTIME_BUILD_DIR:-build-runtime}}"

source build_tools/cmake/setup_build.sh

"${CMAKE_BIN}" -B "${BUILD_DIR}" -G Ninja . \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_BUILD_COMPILER=OFF
"${CMAKE_BIN}" --build "${BUILD_DIR}" -- -k 0
