# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:25abf0c27b7bba497fc84d2382449f18c2eb806725d2e79a92a6ee62489535bb
COPY --from=gcr.io/iree-oss/swiftshader@sha256:05d59843bcd48352e4a14e96e9c6845b04d137e1132dc85f0a05fd7e53210263 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
