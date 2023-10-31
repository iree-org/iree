# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:b654dffe5b69d35f3182ffe1a41be98e3f32bc7843b6f10829a8eb2aa6a345ee
COPY --from=gcr.io/iree-oss/swiftshader@sha256:035ac89d3c357787052a836f4cbd227035260c05c95fa9a53d809600c454e819 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
