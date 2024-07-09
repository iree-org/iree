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

if [[ ! -z "${IREE_BUILD_SETUP_PYTHON_VENV:-}" ]]; then
  # We've been instructed to set up a venv.
  echo "Setting up venv at $IREE_BUILD_SETUP_PYTHON_VENV"
  python3 -m venv "$IREE_BUILD_SETUP_PYTHON_VENV"
  if [[ "${OSTYPE}" =~ ^msys ]]; then
    "$IREE_BUILD_SETUP_PYTHON_VENV"/Scripts/activate.bat
  else
    source "$IREE_BUILD_SETUP_PYTHON_VENV/bin/activate"
  fi
  IREE_PYTHON3_EXECUTABLE="$IREE_BUILD_SETUP_PYTHON_VENV/bin/python"
  python -m pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt
else
  # Just use the host python and yolo with what may be installed.
  # In practice, certain callers take care of this themselves, and we trust
  # them to do it right.
  IREE_PYTHON3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE:-$(which python3)}"
fi

if [[ -d "${BUILD_DIR}" ]]; then
  echo "'${BUILD_DIR}' directory already exists. Will use cached results there."
else
  echo "'${BUILD_DIR}' directory does not already exist. Creating a new one."
  mkdir -p "${BUILD_DIR}"
fi
