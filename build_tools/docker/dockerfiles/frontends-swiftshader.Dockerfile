# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:9e44ee3b53345054ac8c29d35ebef5e1dc7d78b41bad48f7dbbeedddb8a4cf3a
COPY --from=gcr.io/iree-oss/swiftshader@sha256:47501aca2c3cee08358d8f8df78f7d7a36101fb9559cfe439285756620087efa \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
