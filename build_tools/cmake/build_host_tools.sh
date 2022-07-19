#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds IREE host tools for use by other builds. Designed for CI, but can be
# run manually. This uses previously cached build results and does not clear
# build directories.

set -xeuo pipefail

BUILD_DIR="${1:-}"
ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
CMAKE_BIN="${CMAKE_BIN:-$(which cmake)}"

cd "${ROOT_DIR?}"

if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${IREE_HOST_BUILD_DIR:-build-host}"
fi

"${CMAKE_BIN?}" --version
ninja --version

# --------------------------------------------------------------------------- #
if [[ -d "${BUILD_DIR}" ]]; then
  echo "${BUILD_DIR} directory already exists. Will use cached results there."
else
  echo "${BUILD_DIR} directory does not already exist. Creating a new one."
  mkdir -p "${BUILD_DIR}"
fi

# Configure, build, install.
"${CMAKE_BIN}" -G Ninja -B "${BUILD_DIR}" \
  -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \
  -DCMAKE_C_COMPILER="${CC:-clang}" \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DIREE_ENABLE_LLD=ON \
  -DIREE_ENABLE_ASSERTIONS=ON \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  "${ROOT_DIR}"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target install -- -k 0
# --------------------------------------------------------------------------- #
