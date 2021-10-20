# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake-bazel-frontends-vulkan@sha256:9894d346c3dfc7ffe0380f12fa541c220ceb77db92d3bbf0bbf17e679e537ecc AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:7f3947c8494c7a7ae92a48a219b21b5e2ef34bff339a893d0254a33f047e9e10 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
