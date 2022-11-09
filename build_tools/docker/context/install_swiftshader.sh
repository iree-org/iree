#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

SWIFTSHADER_COMMIT="$1"
INSTALL_DIR="$2"

apt-get update
# zlib and xcb/shm.h are needed for compiling SwiftShader.
apt-get install -y git zlib1g-dev libxcb-shm0-dev

git clone https://github.com/google/swiftshader
cd swiftshader
git checkout "${SWIFTSHADER_COMMIT?}"
cd ..
# Only build SwiftShader Vulkan ICD.
cmake -S swiftshader/ -B build-swiftshader/ \
    -GNinja \
    -DSWIFTSHADER_BUILD_TESTS=OFF
cmake --build build-swiftshader/ \
    --config Release \
    --target vk_swiftshader
# Copy the ICD JSON and .so to a known place.
cp -rf build-swiftshader/Linux "${INSTALL_DIR}"
# Keep track of the commit we are using.
echo "${SWIFTSHADER_COMMIT?}" > "${INSTALL_DIR}/git-commit"
