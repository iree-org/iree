# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE using CMake and testing IREE with SwiftShader
# Vulkan implementation.

FROM gcr.io/iree-oss/base@sha256:8bd3dfa4e16c232802842c36f1139730e183f4263b709e76fec04df30de930db AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:2c682d5144e8fa6ea012428299d906abbf41fd04b9c7c5865c657c4e7484c6c5 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
