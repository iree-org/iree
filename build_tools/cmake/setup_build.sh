#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Sets up a CMake build. Requires that BUILD_DIR be set in the environment and
# expects to be *sourced* not invoked, since it sets environment variables.

set -euo pipefail

# Check these exist and print the versions for later debugging.
CMAKE_BIN="${CMAKE_BIN:-$(which cmake)}"
"${CMAKE_BIN}" --version
ninja --version
python3 --version

IREE_PYTHON3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE:-$(which python3)}"

if [[ -d "${BUILD_DIR}" ]]; then
  echo "'${BUILD_DIR}' directory already exists. Will use cached results there."
else
  echo "'${BUILD_DIR}' directory does not already exist. Creating a new one."
  mkdir -p "${BUILD_DIR}"
fi
