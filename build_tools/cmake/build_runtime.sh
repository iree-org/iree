#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build IREE's runtime using CMake. The desired build directory can be passed as
# the first argument. Otherwise, it uses the environment variable
# IREE_RUNTIME_BUILD_DIR, defaulting to "build-runtime". Designed for CI, but
# can be run manually. This reuses the build directory if it already exists.

set -euo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";

BUILD_DIR="${1:-${IREE_RUNTIME_BUILD_DIR:-build-runtime}}"

source "${SCRIPT_DIR}/setup_build.sh"

"${CMAKE_BIN}" -B "${BUILD_DIR}" -G Ninja . \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_BUILD_COMPILER=OFF
"${CMAKE_BIN}" --build "${BUILD_DIR}" -- -k 0
