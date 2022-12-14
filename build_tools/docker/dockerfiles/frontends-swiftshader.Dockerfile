# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:423db6cc3ad8013be06571329786c7fc9dc6c6cc8bc2b1c500615dac22f77899
COPY --from=gcr.io/iree-oss/swiftshader@sha256:9edddda9d84a7d6a1901fbe0cfd45daaa4b0229ca5822ac43c71be69784eab1c \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
