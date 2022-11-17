# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:3b2853c008157829d1c733f3bb194295b667a01ee5ab63d0ee8c4fe246868979
COPY --from=gcr.io/iree-oss/swiftshader@sha256:34a41a491204f7e17caa18bb7fd177f9311aef276af5adcd04e74e84b779c120 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
