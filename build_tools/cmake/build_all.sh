# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build "all" of the IREE project. Designed for CI, but can be run locally.

set -xeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}"

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
BUILD_DIR="${1:-${IREE_BUILD_DIR:-build}}"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"

"$CMAKE_BIN" --version
ninja --version

if [[ -d "${BUILD_DIR}" ]]; then
  echo "Build directory '${BUILD_DIR}' already exists. Will use cached results there."
else
  echo "Build directory '${BUILD_DIR}' does not already exist. Creating a new one."
  mkdir "${BUILD_DIR}"
fi

declare -a CMAKE_ARGS=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"
  # Let's make linking fast
  "-DIREE_ENABLE_LLD=ON"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"

  # Enable docs build on the CI. The additional builds are pretty fast and
  # give us early warnings for some types of website publication errors.
  "-DIREE_BUILD_DOCS=ON"

  # Enable building the python bindings on CI.
  "-DIREE_BUILD_PYTHON_BINDINGS=ON"

  # Enable assertions. We wouldn't want to be testing *only* with assertions
  # enabled, but at the moment only certain CI builds are using this script,
  # e.g. ASan builds are not using this, so by enabling assertions here, we
  # get a reasonable mix of {builds with asserts, builds with other features
  # such as ASan but without asserts}.
  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"

  # Enable CUDA compiler and runtime builds unconditionally. Our CI images all
  # have enough deps to at least build CUDA support and compile CUDA binaries
  # (but not necessarily test on real hardware).
  "-DIREE_HAL_DRIVER_CUDA=ON"
  "-DIREE_TARGET_BACKEND_CUDA=ON"
  "${ROOT_DIR}"
)

"$CMAKE_BIN" "${CMAKE_ARGS[@]}"
echo "Building all"
echo "------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" -- -k 0

echo "Building test deps"
echo "------------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" --target iree-test-deps -- -k 0
