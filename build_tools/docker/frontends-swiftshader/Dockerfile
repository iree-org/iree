# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:1c085c79f1981ec23d4faef56555d2e3adc6488ac355a5e7eb577d7fff4b105e
COPY --from=gcr.io/iree-oss/swiftshader@sha256:3976e79c31b15f58d6b5b9e46b47e97a6d572ceb133385568b6c999bf8e2ea0d \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
