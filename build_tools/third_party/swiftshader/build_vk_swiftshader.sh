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

# Call this script **from the project root** to build SwiftShader's Vulkan ICD:
#   $ bash build_tools/third_party/swiftshader/build_vk_swiftshader.sh
#
# Note that you will need a working CMake installation for this script to
# succeed. On Windows, Visual Studio 2019 is recommended.
#
# Afterwards, set the VK_ICD_FILENAMES environment variable to the absolute
# path of the corresponding vk_swiftshader_icd.json manifest file so the Vulkan
# loader on your system loads it, for example:
#   Windows
#   $ set VK_ICD_FILENAMES=C:\dev\iree\build-swiftshader\Windows\vk_swiftshader_icd.json
#
#   Linux
#   $ VK_ICD_FILENAMES=/dev/iree/build-swiftshader/Linux/vk_swiftshader_icd.json
#
# See https://vulkan.lunarg.com/doc/view/1.1.70.1/windows/loader_and_layer_interface.html
# for further details about the Vulkan loader and ICDs.

# Check that we're in the project root so our relative paths work as expected.
if [[ $(basename "$PWD") != "iree" ]]; then
    >&2 echo "******************************************************"
    >&2 echo "* This script should be run from IREE's project root *"
    >&2 echo "******************************************************"
    exit 1
fi

# Generate build system in build-swiftshader/ for third_party/swiftshader/.
#
# Options:
#   - 64 bit platform and host compiler
#   - Build Vulkan only, don't build GL
#   - Don't build samples or tests
cmake -B build-swiftshader/ \
    -GNinja \
    -DSWIFTSHADER_BUILD_VULKAN=ON \
    -DSWIFTSHADER_BUILD_EGL=OFF \
    -DSWIFTSHADER_BUILD_GLESv2=OFF \
    -DSWIFTSHADER_BUILD_GLES_CM=OFF \
    -DSWIFTSHADER_BUILD_PVR=OFF \
    -DSWIFTSHADER_BUILD_TESTS=OFF \
    third_party/swiftshader/

# Build the project, choosing just the vk_swiftshader target.
cmake --build build-swiftshader/ --config Release --target vk_swiftshader

# Outputs if successful:
#   Linux:   build-swiftshader/Linux/libvk_swiftshader.so
#   Windows: build-swiftshader/Windows/vk_swiftshader.dll
