# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:95c23cc732e4f5b0b005816108c65cea7b024f186be9cdd22defdd6e68384298
COPY --from=gcr.io/iree-oss/swiftshader@sha256:c9be5cbc8467499ae71ec80f3af87b72e746e8903cd52c0be9bb5f7261acc521 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
