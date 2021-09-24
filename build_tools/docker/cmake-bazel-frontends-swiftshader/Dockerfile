# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake-bazel-frontends-vulkan@sha256:46627a46b1366be665ab2e518ad6b6f79594373db0da5af414f290f51442a45f AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:54265ebefcd6464ade8a6b36a3105f6219a2bb804c4a3b2052880fd4213d72a3 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
