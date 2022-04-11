# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:080d9f833bc772b53add2f4ad4ba06723cfb56a05f3a47e12e08c3de95308576
COPY --from=gcr.io/iree-oss/swiftshader@sha256:bb6192ba3d2b156ce4ccdc3aa879769766e0c687c876e1cf93492c0b006d7ac1 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
