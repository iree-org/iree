# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake-bazel-frontends-vulkan@sha256:9dbc756493624aaf82fb492ab12708100bab4d510e3eb4799cfcd6ad08988f4d AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:2c682d5144e8fa6ea012428299d906abbf41fd04b9c7c5865c657c4e7484c6c5 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
