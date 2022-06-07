#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build using CMake, with ThreadSanitizer instrumentation.
# The desired build directory can be passed as
# the first argument. Otherwise, it uses the environment variable
# IREE_TSAN_BUILD_DIR, defaulting to "build-tsan". Designed for CI, but
# can be run manually. This reuses the build directory if it already exists.

set -euo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";

BUILD_DIR="${1:-}"

if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${IREE_TSAN_BUILD_DIR:-build-tsan}"
fi

source "${SCRIPT_DIR}/setup_build.sh"

# IREE_BYTECODE_MODULE_FORCE_SYSTEM_DYLIB_LINKER=ON because TSan doesn't work
# with embedded linker.
# IREE_BUILD_SAMPLES=OFF because samples assume embedded linker.
"${CMAKE_BIN}" -B "${BUILD_DIR}" -G Ninja . \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_ENABLE_TSAN=ON \
  -DIREE_BYTECODE_MODULE_ENABLE_TSAN=ON \
  -DIREE_BYTECODE_MODULE_FORCE_SYSTEM_DYLIB_LINKER=ON

"${CMAKE_BIN}" --build "${BUILD_DIR}" -- -k 0
