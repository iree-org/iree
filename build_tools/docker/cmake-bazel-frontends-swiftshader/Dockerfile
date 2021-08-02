# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake-bazel-frontends-vulkan@sha256:6384367aeb40c4ad18c5620c1663fb0c9f6210a3506e77bd85d43d937cf2fbee AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:ccae32c83c89a31e8fc5542e480c29f28bbf4a3b3b80198c06b687a92c6813f3 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
