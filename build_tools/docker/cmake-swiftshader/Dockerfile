# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE using CMake and testing IREE with SwiftShader
# Vulkan implementation.

FROM gcr.io/iree-oss/cmake-vulkan@sha256:a257e4010d5bd4af19a8d93852b055af51eda3298a6c9d4114c23ff35ccf4b2c AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:7f3947c8494c7a7ae92a48a219b21b5e2ef34bff339a893d0254a33f047e9e10 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
