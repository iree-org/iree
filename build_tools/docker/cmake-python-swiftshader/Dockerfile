# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE using CMake and testing IREE with SwiftShader
# Vulkan implementation.

FROM gcr.io/iree-oss/cmake-python-vulkan@sha256:b1101d7c049aec0add21c94de37780657ac7f6e5b3370d16e86f06040a660582 AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:ccae32c83c89a31e8fc5542e480c29f28bbf4a3b3b80198c06b687a92c6813f3 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
