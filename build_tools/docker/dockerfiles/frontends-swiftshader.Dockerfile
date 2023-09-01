# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:7ecfdda9ef9f64cfa12b1ed203992abab8057ba350ec8c2b7bf63d7dd8f160fc
COPY --from=gcr.io/iree-oss/swiftshader@sha256:e9501304b0a19acfae4a1295dca78f24c839975f25e3c2d434b92c0b5e5f5c5d \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
