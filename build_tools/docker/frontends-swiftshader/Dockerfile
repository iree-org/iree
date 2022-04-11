# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:080d9f833bc772b53add2f4ad4ba06723cfb56a05f3a47e12e08c3de95308576
COPY --from=gcr.io/iree-oss/swiftshader@sha256:c1916e21f0eb7135b92bb4b5d5a3731c23445e91832ff8589a1098f023bb9554 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
