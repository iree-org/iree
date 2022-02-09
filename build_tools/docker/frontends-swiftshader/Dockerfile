# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:3eab65512589e7dabeced8aeb2c392e82f5bf1caafad7639e1b679be908ceb33
COPY --from=gcr.io/iree-oss/swiftshader@sha256:7c030be534f1dfd105a82d7ba31cf0500c80fb60f34ca1cf683cc08cb0e7e46c \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
