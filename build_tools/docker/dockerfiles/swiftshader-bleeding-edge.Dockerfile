# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/base-bleeding-edge@sha256:303e3cc0e3f742b840356ddd61ca58b15a415cbf16e211a972df0c43e901d55f AS install-swiftshader
WORKDIR /install-swiftshader

ARG SWIFTSHADER_COMMIT=d15c42482560fba311e3cac90203438ad972df55

COPY build_tools/docker/context/install_swiftshader.sh ./
RUN ./install_swiftshader.sh "${SWIFTSHADER_COMMIT}" "/swiftshader"

FROM gcr.io/iree-oss/base-bleeding-edge@sha256:303e3cc0e3f742b840356ddd61ca58b15a415cbf16e211a972df0c43e901d55f AS final
COPY --from=install-swiftshader /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
