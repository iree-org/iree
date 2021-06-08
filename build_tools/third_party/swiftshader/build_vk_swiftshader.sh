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
# The parent directory for the installation can be overridden using the first
# positional argument:
#
#   bash build_tools/third_party/swiftshader/build_vk_swiftshader.sh <parent-dir>
#
# If the `.swiftshader` installation dir already exists, it will be deleted and
# rebuilt.
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

set +e  # Ignore errors if not found.
CYGPATH="$(which cygpath 2>/dev/null)"
set -e

if [[ -z "${CYGPATH?}" ]]; then
  # Anything that isn't Windows.
  BASE_DIR="${1:-${HOME?}}"
  SWIFTSHADER_INSTALL_DIR="${BASE_DIR?}/.swiftshader"
else
  # Windows.
  BASE_DIR="${1:-${USERPROFILE?}}"
  SWIFTSHADER_INSTALL_DIR="${BASE_DIR?}"'\.swiftshader'
fi

SWIFTSHADER_COMMIT=84f5eeb6dd9b225f465f93737fa76aad7de355cf
SWIFTSHADER_DIR="$(mktemp --directory --tmpdir swiftshader_XXXXXX)"

#  Clone swiftshader and checkout the appropriate commit.
git clone https://github.com/google/swiftshader "${SWIFTSHADER_DIR?}"
cd "${SWIFTSHADER_DIR?}"
git pull origin master --ff-only
git checkout "${SWIFTSHADER_COMMIT?}"

# Install swiftshader in SWIFTSHADER_INSTALL_DIR.
# Options:
#   - 64 bit platform and host compiler
#   - Build Vulkan only, don't build GL
#   - Don't build samples or tests

echo "Installing to ${SWIFTSHADER_INSTALL_DIR?}"
if [[ -d "${SWIFTSHADER_INSTALL_DIR?}" ]]; then
  echo "  Install directory already exists, cleaning it"
  rm -rf "${SWIFTSHADER_INSTALL_DIR?}"
fi

cmake -B "${SWIFTSHADER_INSTALL_DIR?}" \
    -GNinja \
    -DSWIFTSHADER_BUILD_VULKAN=ON \
    -DSWIFTSHADER_BUILD_EGL=OFF \
    -DSWIFTSHADER_BUILD_GLESv2=OFF \
    -DSWIFTSHADER_BUILD_GLES_CM=OFF \
    -DSWIFTSHADER_BUILD_PVR=OFF \
    -DSWIFTSHADER_BUILD_TESTS=OFF \
    "${SWIFTSHADER_DIR?}"

# Build the project, choosing just the vk_swiftshader target.
cmake --build "${SWIFTSHADER_INSTALL_DIR?}" --config Release --target vk_swiftshader

echo
echo "Ensure the following variable is set in your enviroment:"
if [[ -d "${SWIFTSHADER_INSTALL_DIR?}/Linux/" ]]; then
  echo "  export VK_ICD_FILENAMES=${SWIFTSHADER_INSTALL_DIR?}/Linux/vk_swiftshader_icd.json"
else
  echo '  set VK_ICD_FILENAMES='"${SWIFTSHADER_INSTALL_DIR?}"'\Windows\vk_swiftshader_icd.json'
fi
