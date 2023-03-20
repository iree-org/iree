# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:8f9c268fd4b57a1818695cba9e794dc5dff32b432c36af8345dc1c8f12f0a1e9
COPY --from=gcr.io/iree-oss/swiftshader@sha256:05d59843bcd48352e4a14e96e9c6845b04d137e1132dc85f0a05fd7e53210263 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
