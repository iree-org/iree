# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake-bazel-frontends-vulkan@sha256:cdf41d7ee7707eb3e79d56f2f1f8bd7e9a0ac3a1122dc4f89f8190154796a6bc AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:3cc8c95c6607891a77dbd01ec06ee4f4bfad26f6ff23a368e44fb6189f51ec02 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
