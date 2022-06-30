# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:908d7e0ba3e3a6d448b3c07b39cc0f786a120dd2ced8653993f62fa933e882f7
COPY --from=gcr.io/iree-oss/swiftshader@sha256:542de3bac7f6173add3651f75c4c681f4a8e2d23815332dd82afe85c7d598445 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
