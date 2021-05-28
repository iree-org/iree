# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake-bazel-frontends-vulkan@sha256:fea4d99f19890a8e53adc2c0ca33a2ef845cdad5bcaae96224bc7d783ec83d19 AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:ccae32c83c89a31e8fc5542e480c29f28bbf4a3b3b80198c06b687a92c6813f3 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
