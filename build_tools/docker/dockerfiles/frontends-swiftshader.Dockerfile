# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:e70c8a13ce02fed19fd9407517d8f945070ecd2e2a97657d4d3117ac6d7e0030
COPY --from=gcr.io/iree-oss/swiftshader@sha256:27b843417fd548fab1b636d2b65d2b8b8c09f42a65cd5dde7ecfb9881e4332ac \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
