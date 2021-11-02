# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:f9490f31ff09c2053a9972a002f6af65fcace41d834583754ea93c6df35ff1f4
COPY --from=gcr.io/iree-oss/swiftshader@sha256:715bdde46dbf454c8b20b9f193ff5ca4b02c1e33a1bfbe83d60921aa21d64664 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
