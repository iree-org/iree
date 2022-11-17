# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:d39e98aecffaeeda7ebe9114ce6076b5e9ca7a84dc6608eaf82557d03dd38d90
COPY --from=gcr.io/iree-oss/swiftshader@sha256:484720d57fa816280d9a7455d16e1f9f3d90cc3efdb744646016503ea5ab2b30 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
