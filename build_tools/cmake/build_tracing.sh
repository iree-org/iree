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

ROOT_DIR=$(git rev-parse --show-toplevel)
BUILD_DIR="${1:-${IREE_TRACING_BUILD_DIR:-build-tracing}}"

cd "${ROOT_DIR}"
source "${ROOT_DIR}/build_tools/cmake/setup_build.sh"

# Note: https://github.com/iree-org/iree/issues/6404 prevents us from building
# tests with these other settings. Many tests invoke the compiler tools with
# MLIR threading enabled, which crashes with compiler tracing enabled.
"${CMAKE_BIN?}" -B "${BUILD_DIR}" \
  -G Ninja . \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_ENABLE_RUNTIME_TRACING=ON \
  -DIREE_BUILD_COMPILER=OFF
"${CMAKE_BIN?}" --build "${BUILD_DIR}" -- -k 0
