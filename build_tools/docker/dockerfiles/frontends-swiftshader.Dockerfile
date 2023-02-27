# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:8aab4c5252df3d88aee4a583bc1640c0a1032e92a99d2b92ead057407a82bf6e
COPY --from=gcr.io/iree-oss/swiftshader@sha256:389f877670fd9111601b474b3be7d975b33d837b465262d1318d84f9df6cd193 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
