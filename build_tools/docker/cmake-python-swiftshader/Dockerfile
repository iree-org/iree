# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE using CMake and testing IREE with SwiftShader
# Vulkan implementation.

FROM gcr.io/iree-oss/cmake-python-vulkan@sha256:4b50e3b4e3e11059d9c385d080f3b512fae414c2f5b36ceae72a80c6bfc0cdf1 AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:54265ebefcd6464ade8a6b36a3105f6219a2bb804c4a3b2052880fd4213d72a3 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
