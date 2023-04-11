#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build IREE using CMake with tracing enabled. Designed for CI, but can be run
# manually. This uses previously cached build results and does not clear build
# directories.

set -e
set -x

BUILD_DIR="${1:-${IREE_TRACING_BUILD_DIR:-build-tracing}}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

# Note: https://github.com/openxla/iree/issues/6404 prevents us from building
# tests with these other settings. Many tests invoke the compiler tools with
# MLIR threading enabled, which crashes with compiler tracing enabled.
"${CMAKE_BIN?}" -B "${BUILD_DIR}" \
  -G Ninja . \
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_ENABLE_RUNTIME_TRACING=ON \
  -DIREE_BUILD_COMPILER=OFF
"${CMAKE_BIN?}" --build "${BUILD_DIR}" -- -k 0

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi
