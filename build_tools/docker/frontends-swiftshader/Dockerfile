# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:2ef3b3670808cdaa2a483dddf94fc58d50486148eb73f9f87e8a9d08a628d39a
COPY --from=gcr.io/iree-oss/swiftshader@sha256:8fcbbf19698d831decc902eb1974f58cd76d5841eb404b5a16d90484134ce902 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
