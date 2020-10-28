#!/bin/bash

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

# Call this script to build SwiftShader's Vulkan ICD:
#   $ bash build_tools/third_party/swiftshader/build_vk_swiftshader.sh
#
# The default installation directory is `${HOME?}/.swiftshader`, but it can be
# overridden as follows:
#   $ bash build_tools/third_party/swiftshader/build_vk_swiftshader.sh <swiftshader-install-dir>
#
# Note that you will need a working CMake installation for this script to
# succeed. On Windows, Visual Studio 2019 is recommended.
#
# Afterwards, set the VK_ICD_FILENAMES environment variable to the absolute
# path of the corresponding vk_swiftshader_icd.json manifest file so the Vulkan
# loader on your system loads it.
#
# See https://vulkan.lunarg.com/doc/view/1.1.70.1/windows/loader_and_layer_interface.html
# for further details about the Vulkan loader and ICDs.

SWIFTSHADER_COMMIT=6287c18b1d249152563f0cb2d5cb0c6d0eb9e3d6
SWIFTSHADER_DIR="$(mktemp --directory --tmpdir swiftshader_XXXXXX)"
SWIFTSHADER_INSTALL_DIR="${1:-${HOME?}/.swiftshader}"

# Ensure that we're at the top level iree/ git directory.
IREE_DIR="$(git rev-parse --show-toplevel)"

#  Clone swiftshader and checkout the appropriate commit.
git clone https://github.com/google/swiftshader "${SWIFTSHADER_DIR?}"
cd "${SWIFTSHADER_DIR?}"
git pull origin master --ff-only
git checkout "${SWIFTSHADER_COMMIT?}"
cd "${IREE_DIR?}"

# Install swiftshader in SWIFTSHADER_INSTALL_DIR.
# Options:
#   - 64 bit platform and host compiler
#   - Build Vulkan only, don't build GL
#   - Don't build samples or tests

if [[ -d "${SWIFTSHADER_INSTALL_DIR?}" ]]; then
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
  echo '  $env:VK_ICD_FILENAMES = Resolve-Path' "${SWIFTSHADER_INSTALL_DIR?}/Windows/vk_swiftshader_icd.json"
fi
