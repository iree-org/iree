# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE using CMake and testing IREE with SwiftShader
# Vulkan implementation.

FROM gcr.io/iree-oss/cmake-python-vulkan@sha256:894d91b6ddd3435f0e5cb4424a81296438252dae4d8934e84aa6be4d02b81343 AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:1de855baa93acbc2d4dcbf6c5509ecb6d4fa04ea5f5322a1c4faa54ea278e995 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
