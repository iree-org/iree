# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:bad174c580cdefaf435ce31a7df6bdd7f7cb7bfdcdff5d1acf40f630acf85bf5
COPY --from=gcr.io/iree-oss/swiftshader@sha256:5027d56cdfee743d956bffd035668f7784166a486c48c74b42e5882cb0c289bf \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
