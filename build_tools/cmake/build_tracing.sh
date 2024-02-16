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
# Note: not using ccache since the runtime build should be fast already.

echo "::group::Building with -DIREE_TRACING_PROVIDER=tracy"
"${CMAKE_BIN?}" -B "${BUILD_DIR}" \
  -G Ninja . \
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_ENABLE_RUNTIME_TRACING=ON \
  -DIREE_TRACING_PROVIDER=tracy \
  -DIREE_BUILD_COMPILER=OFF
"${CMAKE_BIN?}" --build "${BUILD_DIR}" -- -k 0
echo "::endgroup::"

echo "::group::Building with -DIREE_TRACING_PROVIDER=console"
"${CMAKE_BIN?}" -B "${BUILD_DIR}" \
  -G Ninja . \
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_ENABLE_RUNTIME_TRACING=ON \
  -DIREE_TRACING_PROVIDER=console \
  -DIREE_BUILD_COMPILER=OFF
"${CMAKE_BIN?}" --build "${BUILD_DIR}" -- -k 0
echo "::endgroup::"
