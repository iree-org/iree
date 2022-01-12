# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:0433f5dfea6f4be6050bb52c3e53c91210aed38244e31e649f65080de2e38fc5
COPY --from=gcr.io/iree-oss/swiftshader@sha256:5d1bb46b68a664ac9c3669a3b4f129d245e0268332f0cd342de4f5b64b9bd253 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
