# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:61a0b0ed212a53d39aa5f78da49c30ce37fc6132c72573e0bbdeb92cc3699803
COPY --from=gcr.io/iree-oss/swiftshader@sha256:a99008c89a1ff024595d8f8a39046b670cdf6dfeac5a6972fdce2dd8934136fa \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
