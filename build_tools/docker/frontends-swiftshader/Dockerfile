# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:6a5e92d8035ecceb59613de888511d13d7452b14fd166c749e1716c421af5ef9
COPY --from=gcr.io/iree-oss/swiftshader@sha256:fe76bc03a3c7a6f08222ad744d973e5738cbac57e6295c476dddbe817b2cc3ec \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
