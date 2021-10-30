# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE using CMake and testing IREE with SwiftShader
# Vulkan implementation.

FROM gcr.io/iree-oss/base@sha256:b8d9863c6ac913f167c6fab319d7cd883ab099312488709ee30b29976d63eb22 AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:2c682d5144e8fa6ea012428299d906abbf41fd04b9c7c5865c657c4e7484c6c5 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
