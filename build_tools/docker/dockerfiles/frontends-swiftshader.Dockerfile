# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:fa8d9a8ff996cc2f98b6bf7103840f20e76763db44b0ac633660f1e063e02693
COPY --from=gcr.io/iree-oss/swiftshader@sha256:ad9d07a63982d4a2c9ada8b5f455b889fed823f0c9c4b28f2548a6ba8cc3d86d \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
