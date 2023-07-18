# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:0073a0ed0b8f9e1978181471155888be33e8d232b223ba317b76301b38c2d51c
COPY --from=gcr.io/iree-oss/swiftshader@sha256:beeb2c9853bb41375f5c9ccfb19b1ed35655c4354cc71fbcf124fe9103a7f543 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
