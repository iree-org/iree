# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake-bazel-frontends-vulkan@sha256:45cd0049355f68975034d6c469a0b1d0c232debe8f8c2ee00002cf605afe8fa7 AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:7f3947c8494c7a7ae92a48a219b21b5e2ef34bff339a893d0254a33f047e9e10 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
