#!/bin/bash

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script builds SwiftShader's Vulkan ICD. By default, it creates a
# `.swiftshader` installation directory in your OS's home directory (`HOME` on
# not-windows, `USERPROFILE` on windows):
#
#   bash build_tools/third_party/swiftshader/build_vk_swiftshader.sh
#
# The installation directory can be overridden using the first positional
# argument:
#
#   bash build_tools/third_party/swiftshader/build_vk_swiftshader.sh <parent-dir>
#
# If the installation dir already exists, it will be deleted and rebuilt.
#
# Note that you will need a working CMake installation for this script to
# succeed. On Windows, Visual Studio 2019 is recommended.
#
# Afterward, you'll need to set the `VK_ICD_FILENAMES` environment variable to
# the absolute path of the `vk_swiftshader_icd.json` manifest file. This tells
# the Vulkan loader on your system to load it. Assuming you use the default
# installation directory this can be done on not-Windows via:
#
#   export VK_ICD_FILENAMES="${HOME?}/.swiftshader/Linux/vk_swiftshader_icd.json"
#
# or on Windows via:
#
#   set VK_ICD_FILENAMES=%USERPROFILE%\.swiftshader\Windows\vk_swiftshader_icd.json
#
# If you used a custom installation directory then the correct path will be
# printed to stdout.
#
# See https://vulkan.lunarg.com/doc/view/1.1.70.1/windows/loader_and_layer_interface.html
# for further details about the Vulkan loader and ICDs.

set -euo pipefail

set +e  # Ignore errors if not found.
CYGPATH="$(which cygpath 2>/dev/null)"
set -e

# This isn't an argument so that we can set the swiftshader commit in one place
# for the whole project.
SWIFTSHADER_COMMIT=32f9332d1d7aacbdba7c1aa5df894bb1890bb2cc
SWIFTSHADER_SRC_DIR="$(mktemp --directory --tmpdir swiftshader_src_XXXXXX)"
SWIFTSHADER_BUILD_DIR="$(mktemp --directory --tmpdir swiftshader_build_XXXXXX)"
CC=${CC:-clang}
CXX=${CXX:-clang++}

if [[ -z "${CYGPATH}" ]]; then
  # Anything that isn't Windows.
  SWIFTSHADER_INSTALL_DIR="${1:-${HOME}/.swiftshader}"
  PLATFORM_DIR="${SWIFTSHADER_BUILD_DIR}/Linux"
  SET_VK_ICD_CMD="export VK_ICD_FILENAMES=${SWIFTSHADER_INSTALL_DIR}/vk_swiftshader_icd.json"
else
  # Windows.
  SWIFTSHADER_INSTALL_DIR="${1:-${USERPROFILE}\.swiftshader}"
  PLATFORM_DIR="${SWIFTSHADER_BUILD_DIR}"'\Windows\'
  SET_VK_ICD_CMD='set VK_ICD_FILENAMES='"${SWIFTSHADER_INSTALL_DIR}"'\vk_swiftshader_icd.json'
fi

#  Shallow clone the specific swiftshader commit.
cd "${SWIFTSHADER_SRC_DIR}"
git init --quiet
git remote add origin https://github.com/google/swiftshader
git fetch --depth 1 origin "${SWIFTSHADER_COMMIT}"
git -c advice.detachedHead=false checkout "${SWIFTSHADER_COMMIT}"

# Build in a temporary directory
cmake -B "${SWIFTSHADER_BUILD_DIR}" \
      -GNinja \
      -DCMAKE_C_COMPILER=${CC} \
      -DCMAKE_CXX_COMPILER=${CXX} \
    -DSWIFTSHADER_BUILD_TESTS=OFF \
    -DSWIFTSHADER_WARNINGS_AS_ERRORS=OFF \
    .
# Build the project, choosing just the vk_swiftshader target.
cmake --build "${SWIFTSHADER_BUILD_DIR}" --config Release --target vk_swiftshader -- -k 0


echo "Installing to ${SWIFTSHADER_INSTALL_DIR}"
if [[ -d "${SWIFTSHADER_INSTALL_DIR}" ]]; then
  echo "  Install directory already exists, cleaning it"
  rm -rf "${SWIFTSHADER_INSTALL_DIR}"
fi

if ! [[ -d "${PLATFORM_DIR}" ]]; then
  echo "Build failed to create expected directory '${PLATFORM_DIR}'" >&2
  exit 1
fi

# Copy the actual built artifacts into the installation directory
cp -rf "${PLATFORM_DIR}" "${SWIFTSHADER_INSTALL_DIR}"

# Keep track of the commit we are using.
echo "${SWIFTSHADER_COMMIT}" > "${SWIFTSHADER_INSTALL_DIR}/git-commit"

# Cleanup
rm -rf "${SWIFTSHADER_SRC_DIR}" "${SWIFTSHADER_BUILD_DIR}"

echo
echo "Ensure the following variable is set in your enviroment:"
echo "  " "${SET_VK_ICD_CMD}"
