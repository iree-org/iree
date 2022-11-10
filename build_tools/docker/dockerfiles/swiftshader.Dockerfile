# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/base@sha256:7c3027c48b94fc38e64488987fc7893c100526c57308d25cef0c6b76a2dfe117 AS install-swiftshader
WORKDIR /install-swiftshader

ARG SWIFTSHADER_COMMIT=d15c42482560fba311e3cac90203438ad972df55

COPY build_tools/docker/context/install_swiftshader.sh ./
RUN ./install_swiftshader.sh "${SWIFTSHADER_COMMIT}" "/swiftshader"

FROM gcr.io/iree-oss/base@sha256:7c3027c48b94fc38e64488987fc7893c100526c57308d25cef0c6b76a2dfe117 AS final
COPY --from=install-swiftshader /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
