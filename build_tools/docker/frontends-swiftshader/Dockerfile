# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:65c2eeab850e274dfa0ec0312b755061c95c5922dd17186d4d42df89f79d28f3
COPY --from=gcr.io/iree-oss/swiftshader@sha256:d27bd8ffdffa7abcd02970408b2665b2a7b1e8238e5b3561672633bd3e52652e \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
