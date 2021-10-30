# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake-bazel-frontends-vulkan@sha256:079bd1444c98932cc8b54a8d469f444d982e94a176ee3701da4df45dd94a5e59 AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:2c682d5144e8fa6ea012428299d906abbf41fd04b9c7c5865c657c4e7484c6c5 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
