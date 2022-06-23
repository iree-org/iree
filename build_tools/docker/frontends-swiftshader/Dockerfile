# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:a39123add274ffc540bfe7937c95e0d85a2e321e3648c98ec033b8f38a84011a
COPY --from=gcr.io/iree-oss/swiftshader@sha256:face4d700d4cbd5ad2d319da28ee7c79a614091498014c3dc62d422dbad0f2a7 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
