# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:6302f27bd4fe35c7244fb63cbd8dbb118d36f1aefcc84babd328b3224da47d4a
COPY --from=gcr.io/iree-oss/swiftshader@sha256:066672cc54693e3ab7d557521cf1dcb4fab8ea839262470650b85bf27696de4b \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
