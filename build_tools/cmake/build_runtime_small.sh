#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build IREE's runtime using CMake. Designed for CI, but can be run manually.
# This uses previously cached build results and does not clear build
# directories.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_RUNTIME_SMALL_BUILD_DIR:-build-runtime-small}}"

source build_tools/cmake/setup_build.sh
# Note: not using ccache since the runtime build should be fast already.

"${CMAKE_BIN?}" -B "${BUILD_DIR}" \
  -G Ninja . \
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DIREE_SIZE_OPTIMIZED=ON \
  -DIREE_BUILD_COMPILER=OFF
"${CMAKE_BIN?}" --build "${BUILD_DIR}" -- -k 0
