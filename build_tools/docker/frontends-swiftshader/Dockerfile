# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:e7faf90e0f77ad6f9236df88e7af26447644c0860610c2cb7a3202c2b961795f
COPY --from=gcr.io/iree-oss/swiftshader@sha256:2fc835bd93f2fb5ec763b882bb697e5293c24b071f9e4c3e6c0ee5d290c3d347 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
