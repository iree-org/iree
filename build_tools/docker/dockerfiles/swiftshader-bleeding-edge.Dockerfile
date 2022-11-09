# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/base-bleeding-edge@sha256:9b496d8ed27c083f63dae68f6cda2af6a711d54ad496f9b1064b3dd2e2e3e6f4 AS install-swiftshader
WORKDIR /install-swiftshader

ARG SWIFTSHADER_COMMIT=d15c42482560fba311e3cac90203438ad972df55

COPY build_tools/docker/context/install_swiftshader.sh ./
RUN ./install_swiftshader.sh "${SWIFTSHADER_COMMIT}" "/swiftshader"

FROM gcr.io/iree-oss/base-bleeding-edge@sha256:9b496d8ed27c083f63dae68f6cda2af6a711d54ad496f9b1064b3dd2e2e3e6f4 AS final
COPY --from=install-swiftshader /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
