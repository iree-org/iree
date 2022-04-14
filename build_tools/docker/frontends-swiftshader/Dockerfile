# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:e0a5d9a71cc7f087f4caaea0cc9b9a86ab4765b217970772780e446b4fdc65b0
COPY --from=gcr.io/iree-oss/swiftshader@sha256:7e397b1cf65f65a9b38dd5b03721387a0fb2953350032a71a5a3ed632d497362 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
