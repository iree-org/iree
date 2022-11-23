# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:4b8f97bccf8443f0314d25c2737a46c979d9acffd28ad52f7d0ce5b879449da5
COPY --from=gcr.io/iree-oss/swiftshader@sha256:33368be42c7de747a234c5192b993436ceff08901250724c24af02d31e817fb8 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
