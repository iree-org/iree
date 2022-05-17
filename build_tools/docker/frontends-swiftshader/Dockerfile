# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:5868b0de784a590d710c837c3799d9d52456510e6db3d555661ba5f9f87ce213
COPY --from=gcr.io/iree-oss/swiftshader@sha256:88ac65bb86014b131d4081c456e480453dce2ce3b5de3319193fb698596df0ce \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
