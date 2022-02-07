# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:c720dc8788a49c5a3cbd6b947cd37360ac9160dac043170dd513c8eafb3c5818
COPY --from=gcr.io/iree-oss/swiftshader@sha256:9a83eb64f53c354772fbe53d4de6669eea7acb39d373b134f66b17fbdce22936 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
